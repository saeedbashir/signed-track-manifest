"""One title end to end: analyse, choose the signer, cut the layers, write the manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stm.model import (
    Caption,
    CaptionKind,
    Catalogue,
    CatalogueEntry,
    Extraction,
    ExtractionMethod,
    FillMethod,
    Provenance,
    Rect,
    SignTrack,
    Source,
    SourceLayout,
    TitleManifest,
    VideoAsset,
)
from stm.schema import is_valid, validate_title
from stm_pipeline import ffmpeg
from stm_pipeline.activity import ActivityMeasurement, measure_activity
from stm_pipeline.align import align, to_webvtt
from stm_pipeline.analyze import Analysis, analyze
from stm_pipeline.asr import DEFAULT_MODEL, transcribe_cached
from stm_pipeline.captions import read_as_vtt
from stm_pipeline.confidence import confidence as confidence_of
from stm_pipeline.config import PipelineConfig
from stm_pipeline.crop import (
    Edge,
    Layout,
    classify_layout,
    fixed_rect,
    grow_window,
    pad_rect,
    scale_rect,
)
from stm_pipeline.detect.base import Detector
from stm_pipeline.identify import ClusterFeatures, choose_signer, cluster_detections, rank_clusters
from stm_pipeline.ingest import SourceEntry, fetch_captions, fetch_media
from stm_pipeline.official_report import fetch_official_report, parse_official_report
from stm_pipeline.simplify import NullSimplifier, Simplifier


class NoSignerFoundError(RuntimeError):
    """No detected cluster met the eligibility rules."""


@dataclass(frozen=True)
class Decision:
    """Everything the pipeline decided about where the signer is, before any encoding."""

    rect_source: Rect
    rect_sample: Rect
    person_rect_sample: Rect | None
    layout: Layout
    method: ExtractionMethod
    confidence: float
    chosen: ClusterFeatures | None
    ranked: list[ClusterFeatures]
    analysis: Analysis | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rectSource": self.rect_source.to_dict(),
            "rectSample": self.rect_sample.to_dict(),
            "personRectSample": (
                self.person_rect_sample.to_dict() if self.person_rect_sample else None
            ),
            "layout": str(self.layout.kind),
            "edge": self.layout.edge,
            "method": str(self.method),
            "confidence": round(self.confidence, 4),
            "chosenClusterId": self.chosen.cluster_id if self.chosen else None,
            "clusters": [f.to_dict() for f in self.ranked],
            "sampledFrames": self.analysis.sampled_count if self.analysis else None,
            "detector": self.analysis.detector_name if self.analysis else None,
            "sampleScale": self.analysis.sample_scale if self.analysis else None,
        }


def decide(
    video: Path, detector: Detector, config: PipelineConfig, manual: SourceEntry | None = None
) -> Decision:
    """Locate the signer. With a manual rect, skip detection and trust the human."""
    if manual is not None and manual.crop_rect is not None:
        info = ffmpeg.probe(video)
        layout_kind = manual.layout or SourceLayout.CORNER_INSET
        rect = manual.crop_rect
        edge: Edge | None = None
        panel: Rect | None = None
        if layout_kind is SourceLayout.SIDE_PANEL:
            edge = "left" if rect.x <= info.width - rect.x2 else "right"
            panel = (
                Rect(0, 0, rect.x2, info.height)
                if edge == "left"
                else Rect(rect.x, 0, info.width - rect.x, info.height)
            )
        return Decision(
            rect_source=rect,
            rect_sample=rect,
            person_rect_sample=None,
            layout=Layout(layout_kind, edge, panel),
            method=ExtractionMethod.MANUAL_CROP,
            confidence=1.0,
            chosen=None,
            ranked=[],
            analysis=None,
        )

    analysis = analyze(video, detector, config)
    clusters = cluster_detections(analysis.frames, config)
    ranked = rank_clusters(
        clusters, analysis.sampled_count, analysis.sample_width, analysis.sample_height, config
    )
    chosen = choose_signer(ranked)
    if chosen is None:
        raise NoSignerFoundError(
            "no cluster was persistent and small enough to be the interpreter; "
            f"{len(ranked)} clusters seen"
        )
    boxes = clusters[chosen.cluster_id].boxes
    rect_sample = fixed_rect(
        boxes,
        analysis.sample_width,
        analysis.sample_height,
        config.crop_low_percentile,
        config.crop_high_percentile,
        config.crop_pad,
    )
    layout = classify_layout(
        rect_sample,
        analysis.sample_width,
        analysis.sample_height,
        analysis.temporal_var,
        analysis.mean_gray,
        config,
    )
    person_rect = rect_sample
    if config.window_growth:
        rect_sample = grow_window(
            rect_sample,
            analysis.temporal_var,
            analysis.mean_gray,
            analysis.sample_width,
            analysis.sample_height,
            config,
        )
        rect_sample = pad_rect(
            rect_sample, config.window_margin, analysis.sample_width, analysis.sample_height
        )
        # Classify on the window, not on the person standing inside it.
        layout = classify_layout(
            rect_sample,
            analysis.sample_width,
            analysis.sample_height,
            analysis.temporal_var,
            analysis.mean_gray,
            config,
        )
    inv = 1.0 / analysis.sample_scale if analysis.sample_scale else 1.0
    rect_source = scale_rect(rect_sample, inv, analysis.info.width, analysis.info.height)
    layout_source = layout
    if layout.panel is not None:
        layout_source = Layout(
            layout.kind,
            layout.edge,
            scale_rect(layout.panel, inv, analysis.info.width, analysis.info.height),
        )
    return Decision(
        rect_source=rect_source,
        rect_sample=rect_sample,
        person_rect_sample=person_rect,
        layout=layout_source,
        method=ExtractionMethod.DETECTED_CROP,
        confidence=confidence_of(chosen),
        chosen=chosen,
        ranked=ranked,
        analysis=analysis,
    )


def process_title(
    entry: SourceEntry,
    detector: Detector,
    config: PipelineConfig,
    out_dir: Path,
    work_dir: Path,
    simplifier: Simplifier | None = None,
) -> Path:
    """Produce ``out_dir/<id>/`` with main.mp4, signer-<lang>.mp4, poster.jpg, captions and
    manifest.json. Returns the manifest path. Raises if the manifest fails validation."""
    simplifier = simplifier or NullSimplifier()
    title_dir = out_dir / entry.id
    title_dir.mkdir(parents=True, exist_ok=True)
    (title_dir / "captions").mkdir(exist_ok=True)

    video = fetch_media(entry, work_dir)
    info = ffmpeg.probe(video)
    decision = decide(video, detector, config, manual=entry)

    # Activity is measured on the decided rectangle, for hand-measured and
    # detected crops alike — the manual path never ran analysis, so this is
    # the only pass that sees how the interpreter moves over time.
    measurement: ActivityMeasurement | None = None
    if config.activity:
        measurement = measure_activity(video, decision.rect_source, config)

    report = decision.to_dict()
    report["activity"] = measurement.to_dict() if measurement else None
    (title_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Signer layer.
    signer_name = f"signer-{entry.sign_language}.mp4"
    signer_path = title_dir / signer_name
    ffmpeg.crop(
        video, decision.rect_source, signer_path, config.signer_max_height, config.signer_crf
    )
    signer_info = ffmpeg.probe(signer_path)

    # Main layer.
    main_path = title_dir / "main.mp4"
    keep = decision.layout.main_keep_rect(info.width, info.height)
    if keep is not None:
        ffmpeg.crop_away(video, keep, main_path, config.main_crf)
        fill_method = FillMethod.NONE
    else:
        fill_method = config.fill_method
        ffmpeg.fill(
            video,
            decision.rect_source,
            main_path,
            info.width,
            info.height,
            fill_method,
            config.fill_blur_sigma,
            config.main_crf,
        )
    main_info = ffmpeg.probe(main_path)

    # Poster from the main layer, where no signer is visible.
    poster_path = title_dir / "poster.jpg"
    ffmpeg.poster(main_path, max(0.0, info.duration_s * config.poster_at_fraction), poster_path)

    # Captions. A caption file the publisher supplied wins; failing that, a
    # published transcript aligned to the audio — the publisher's words on the
    # recogniser's clock, which is still human text and is labelled as such.
    captions: list[Caption] = []
    vtt: str | None = None
    cap_src = fetch_captions(entry, work_dir)
    if cap_src is not None:
        vtt = read_as_vtt(cap_src)
    elif entry.official_report:
        page = fetch_official_report(entry.official_report, work_dir / entry.id)
        contributions = parse_official_report(page)
        spoken = transcribe_cached(
            video, work_dir / entry.id, DEFAULT_MODEL, entry.caption_language
        )
        cues, summary = align(contributions, spoken)
        report["alignment"] = summary.to_dict()
        # The analysis report stays out of the published title, so this is the
        # only place a Batch log shows how the alignment went.
        print(
            f"  captions: {len(contributions)} contributions, {summary.report_words} words; "
            f"{summary.anchored} anchored ({summary.anchored_fraction:.0%}), "
            f"{summary.events} stage directions, {summary.unheard} unheard; {summary.cues} cues"
        )
        (title_dir / "analysis.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        vtt = to_webvtt(
            cues,
            "Text: the publisher's Official Report, unchanged. "
            "Timing: aligned automatically to the audio by signed-track-manifest.",
        )
    if vtt is not None:
        verbatim_name = f"captions/verbatim.{entry.caption_language}.vtt"
        (title_dir / verbatim_name).write_text(vtt, encoding="utf-8")
        captions.append(
            Caption(kind=CaptionKind.VERBATIM, language=entry.caption_language, url=verbatim_name)
        )
        simplified = simplifier.simplify(vtt)
        if simplified:
            simplified_name = f"captions/simplified.{entry.caption_language}.vtt"
            (title_dir / simplified_name).write_text(simplified, encoding="utf-8")
            captions.append(
                Caption.generated(
                    CaptionKind.SIMPLIFIED, entry.caption_language, simplified_name, simplifier.name
                )
            )

    manifest = TitleManifest(
        id=entry.id,
        title=entry.title,
        duration_ms=info.duration_ms,
        source=Source(
            publisher=entry.publisher,
            url=entry.url,
            licence=entry.licence,
            attribution=entry.attribution,
            production_credit=entry.production_credit,
            acquisition_route=entry.acquisition_route,
        ),
        main=VideoAsset("main.mp4", main_info.width, main_info.height, "h264"),
        poster="poster.jpg",
        sign_language=[
            SignTrack(
                language=entry.sign_language,
                language_label=entry.language_label,
                provenance=Provenance.HUMAN_INTERPRETER,
                extraction=Extraction(
                    method=decision.method,
                    source_layout=decision.layout.kind,
                    confidence=decision.confidence,
                    crop_rect=decision.rect_source,
                    fill_method=fill_method,
                ),
                track=VideoAsset(signer_name, signer_info.width, signer_info.height, "h264"),
                activity=measurement.activity if measurement else None,
            )
        ],
        captions=captions,
    )
    data = manifest.to_dict()
    issues = validate_title(data)
    if not is_valid(issues):
        raise RuntimeError("pipeline produced an invalid manifest:\n" + "\n".join(map(str, issues)))
    manifest_path = title_dir / "manifest.json"
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    return manifest_path


def build_catalogue(manifest_paths: list[Path], out_path: Path) -> Path:
    """Index per-title manifests into a catalogue, with URLs relative to the catalogue file."""
    entries: list[CatalogueEntry] = []

    def _rel(base: Path, ref: str) -> str:
        return ref if "://" in ref else str(base / ref)

    for mp in manifest_paths:
        data = json.loads(mp.read_text(encoding="utf-8"))
        rel_dir = (
            mp.parent.relative_to(out_path.parent)
            if mp.is_relative_to(out_path.parent)
            else mp.parent
        )
        confs = [
            t["extraction"]["confidence"] for t in data.get("signLanguage", []) if "extraction" in t
        ]
        entries.append(
            CatalogueEntry(
                id=data["id"],
                title=data["title"],
                duration_ms=int(data["durationMs"]),
                poster=_rel(rel_dir, data["video"]["poster"]),
                manifest_url=str(rel_dir / mp.name),
                sign_languages=[t["language"] for t in data.get("signLanguage", [])],
                confidence=min(confs) if confs else None,
            )
        )
    out_path.write_text(Catalogue(entries).to_json(), encoding="utf-8")
    return out_path
