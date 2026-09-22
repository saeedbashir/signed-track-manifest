"""``stm``: the pipeline command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from stm.model import FillMethod, Rect
from stm_pipeline import ffmpeg
from stm_pipeline.activity import idle_fraction, measure_activity, sparkline
from stm_pipeline.config import PipelineConfig
from stm_pipeline.detect.base import Detector
from stm_pipeline.detect.models import MODELS, ensure_model
from stm_pipeline.harness.ground_truth import load_ground_truth
from stm_pipeline.harness.score import PASS_IOU, score
from stm_pipeline.ingest import (
    BlockedSourceError,
    SourceError,
    fetch_captions,
    fetch_media,
    load_sources,
)
from stm_pipeline.run import NoSignerFoundError, build_catalogue, process_title
from stm_pipeline.simplify import DEFAULT_MODEL, KNOWN_MODELS, BedrockSimplifier, NullSimplifier
from stm_pipeline.spike import format_cluster_table, spike
from stm_pipeline.triage import format_table, suggest_sources_entry, triage


def _detector(name: str) -> Detector:
    from stm_pipeline.detect.onnx_person import OnnxPersonDetector

    return OnnxPersonDetector(model=name)


def _config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    every = getattr(args, "every", None)
    if every:
        cfg = replace(cfg, sample_every=int(every))
    detector = getattr(args, "detector", None)
    if detector:
        cfg = replace(cfg, detector=str(detector))
    fill = getattr(args, "fill", None)
    if isinstance(fill, str):
        cfg = replace(cfg, fill_method=FillMethod(fill))
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--detector", choices=sorted(MODELS), default="yolox-tiny")
    p.add_argument("--every", type=int, help="sample every Nth frame (default 5)")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stm",
        description=(
            "Signed Track Manifest pipeline: find the interpreter, cut the layers, emit STM."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("models", help="manage detector weights")
    msub = m.add_subparsers(dest="models_cmd", required=True)
    mf = msub.add_parser("fetch", help="download and verify a model")
    mf.add_argument("--model", choices=sorted(MODELS), default="yolox-tiny")
    msub.add_parser("list", help="list known models and their licences")

    i = sub.add_parser("ingest", help="validate sources.yaml and fetch media")
    i.add_argument("sources", type=Path)
    i.add_argument("--work", type=Path, default=Path("work"))
    i.add_argument("--id", dest="only_id", help="only this source id")

    r = sub.add_parser("run", help="process titles end to end")
    r.add_argument("sources", type=Path)
    r.add_argument("--id", dest="only_id", help="only this source id (default: all)")
    r.add_argument("--out", type=Path, default=Path("out"))
    r.add_argument("--work", type=Path, default=Path("work"))
    r.add_argument("--fill", choices=[str(f) for f in FillMethod])
    r.add_argument(
        "--simplify",
        action="store_true",
        help="also generate a simplified caption track with Claude on Amazon Bedrock "
        "(needs AWS credentials and Bedrock model access; always written reviewed:false)",
    )
    r.add_argument(
        "--bedrock-model",
        default=DEFAULT_MODEL,
        help=f"Bedrock model id (default {DEFAULT_MODEL}; known: {', '.join(KNOWN_MODELS)})",
    )
    r.add_argument("--bedrock-region", help="AWS region (default: the usual AWS env/config chain)")
    r.add_argument("--chunk-size", type=int, default=40, help="cues per simplification request")
    _add_common(r)

    s = sub.add_parser("spike", help="watchable outputs for one clip: box overlay, fill preview")
    s.add_argument("clip", type=Path)
    s.add_argument("--out", type=Path, help="output directory (default: next to the clip)")
    s.add_argument(
        "--fill",
        action="append",
        choices=[str(f) for f in FillMethod if f is not FillMethod.NONE],
        help="fill method(s) to preview; repeatable (default: interpolated-surround)",
    )
    s.add_argument("--detections-video", action="store_true", help="also write per-frame boxes")
    _add_common(s)

    h = sub.add_parser("harness", help="ground truth and scoring")
    hsub = h.add_subparsers(dest="harness_cmd", required=True)
    hs = hsub.add_parser("score", help="score the locate step against ground truth")
    hs.add_argument("ground_truth", type=Path)
    hs.add_argument("--threshold", type=float, default=PASS_IOU)
    hs.add_argument("--json", type=Path, help="write the report here")
    hs.add_argument("--strict", action="store_true", help="exit 1 if any clip fails")
    _add_common(hs)
    hl = hsub.add_parser("label", help="draw a ground-truth rectangle for a clip")
    hl.add_argument("clip", type=Path)
    hl.add_argument("--gt", type=Path, default=Path("harness/ground_truth.yaml"))
    hl.add_argument("--at", type=float, default=5.0, help="seconds into the clip")
    hl.add_argument("--face", action="store_true", help="also draw the face rectangle")

    t = sub.add_parser(
        "triage",
        help="rank candidate clips by how good a demo they would make "
        "(layout, extraction confidence, and how hard the fill will be)",
    )
    t.add_argument("clips", nargs="+", type=Path)
    t.add_argument("--json", type=Path, help="write the full report here")
    t.add_argument(
        "--sources",
        action="store_true",
        help="print sources.yaml skeletons with the measurable fields filled in",
    )
    _add_common(t)

    a = sub.add_parser(
        "activity",
        help="measure signing activity inside a rectangle, per second, as the manifest carries it",
    )
    a.add_argument("clip", type=Path)
    a.add_argument("--rect", required=True, help="x,y,w,h in source pixels")
    a.add_argument("--interval-ms", type=int, default=1000)
    a.add_argument("--json", type=Path, help="write the activity object here")
    _add_common(a)

    c = sub.add_parser("catalogue", help="index manifests into a catalogue")
    c.add_argument("manifests", nargs="+", type=Path)
    c.add_argument("-o", "--out", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.cmd == "models":
        if args.models_cmd == "list":
            for spec in MODELS.values():
                print(f"{spec.name:12} {spec.input_size}px  {spec.licence}\n{'':12} {spec.source}")
            return 0
        path = ensure_model(args.model)
        print(f"{args.model}: {path}")
        return 0

    if not ffmpeg.available():
        print(
            "ffmpeg and ffprobe are required on PATH (or set STM_FFMPEG / STM_FFPROBE)",
            file=sys.stderr,
        )
        return 2

    if args.cmd == "ingest":
        try:
            entries = load_sources(args.sources)
            for e in entries:
                if args.only_id and e.id != args.only_id:
                    continue
                video = fetch_media(e, args.work)
                caps = fetch_captions(e, args.work)
                print(f"{e.id}: {video}" + (f"  captions: {caps}" if caps else ""))
        except (SourceError, BlockedSourceError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "run":
        config = _config(args)
        try:
            entries = load_sources(args.sources)
        except SourceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        detector = _detector(config.detector)
        simplifier = (
            BedrockSimplifier(
                model=args.bedrock_model,
                region=args.bedrock_region,
                chunk_size=args.chunk_size,
            )
            if args.simplify
            else NullSimplifier()
        )
        failures = 0
        written: list[Path] = []
        for e in entries:
            if args.only_id and e.id != args.only_id:
                continue
            try:
                mp = process_title(e, detector, config, args.out, args.work, simplifier)
                if isinstance(simplifier, BedrockSimplifier) and simplifier.stats.cues:
                    st = simplifier.stats
                    print(
                        f"  captions: {st.rewritten}/{st.cues} cues simplified "
                        f"({st.chunks} requests, {st.retries} retries, "
                        f"{st.kept_verbatim} kept verbatim)"
                    )
                    for failure in st.failures:
                        print(f"  captions: {failure}", file=sys.stderr)
                written.append(mp)
                print(f"{e.id}: {mp}")
            except (
                NoSignerFoundError,
                SourceError,
                ffmpeg.FfmpegError,
                RuntimeError,
                OSError,
            ) as exc:
                failures += 1
                print(f"{e.id}: FAILED — {exc}", file=sys.stderr)
        if written:
            cat = build_catalogue(written, args.out / "catalogue.json")
            print(f"catalogue: {cat}")
        return 1 if failures else 0

    if args.cmd == "spike":
        config = _config(args)
        detector = _detector(config.detector)
        fills = [FillMethod(f) for f in (args.fill or ["interpolated-surround"])]
        try:
            result = spike(args.clip, detector, config, args.out, fills, args.detections_video)
        except NoSignerFoundError as exc:
            print(f"no signer found: {exc}", file=sys.stderr)
            return 1
        d = result.decision
        print(format_cluster_table(d.ranked, d.chosen.cluster_id if d.chosen else None))
        print(f"\nlayout: {d.layout.kind}  confidence: {d.confidence:.2f}")
        print(f"rect (source px): {json.dumps(d.rect_source.to_dict())}")
        if config.activity:
            m = measure_activity(args.clip, d.rect_source, config)
            print(
                f"activity: |{sparkline(m.activity.values)}|  "
                f"p95 {m.p95:.1f}  idle<20: {idle_fraction(m.activity.values):.0%}"
            )
        for k, v in result.outputs.items():
            print(f"{k:28} {v}")
        return 0

    if args.cmd == "activity":
        config = _config(args)
        try:
            x, y, w, h = (int(v) for v in args.rect.split(","))
            rect = Rect(x, y, w, h)
        except ValueError as exc:
            print(f"error: --rect wants x,y,w,h in source pixels ({exc})", file=sys.stderr)
            return 1
        m = measure_activity(args.clip, rect, config, args.interval_ms)
        print(f"|{sparkline(m.activity.values)}|")
        print(
            f"{len(m.activity.values)} intervals of {m.activity.interval_ms} ms; "
            f"p95 motion {m.p95:.2f} became 100; {m.samples} sampled frames; "
            f"idle (<20): {idle_fraction(m.activity.values):.0%}"
        )
        if args.json:
            args.json.write_text(
                json.dumps(m.activity.to_dict(), separators=(",", ":")) + "\n", encoding="utf-8"
            )
            print(f"wrote {args.json}")
        return 0

    if args.cmd == "harness":
        if args.harness_cmd == "label":
            from stm_pipeline.harness.label import label

            gt = label(args.clip, args.gt, args.at, args.face)
            if gt is None:
                print("cancelled")
                return 1
            print(f"saved {gt.rect.to_dict()} for {args.clip} -> {args.gt}")
            return 0
        config = _config(args)
        items = load_ground_truth(args.ground_truth)
        if not items:
            print(f"{args.ground_truth} has no clips", file=sys.stderr)
            return 1
        report = score(items, _detector(config.detector), config, args.threshold)
        print(report.table())
        if args.json:
            report.write_json(args.json)
        return 1 if args.strict and report.passed < len(report.scores) else 0

    if args.cmd == "triage":
        config = _config(args)
        detector = _detector(config.detector)
        results = [triage(clip, detector, config) for clip in args.clips]
        print(format_table(results))
        if args.sources:
            print(
                "\n# sources.yaml skeletons. Every TODO is a judgement call,"
                " not a measurement.\nsources:"
            )
            for r in sorted(results, key=lambda x: x.rank):
                if r.ok:
                    print(suggest_sources_entry(r))
        if args.json:
            args.json.write_text(
                json.dumps([r.to_dict() for r in results], indent=2) + "\n", encoding="utf-8"
            )
        return 0

    if args.cmd == "catalogue":
        out = build_catalogue(list(args.manifests), args.out)
        print(out)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
