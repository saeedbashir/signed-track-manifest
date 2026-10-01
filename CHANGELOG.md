# Changelog

Format versions and library releases. The format version is `stmVersion` in a
manifest; the library version is `pyproject.toml`.

## Unreleased

Library changes; the format is still 0.2 and no schema changed.

- **Added** caption tracks from a publisher's written transcript.
  `stm captions align VIDEO --report URL|FILE` (and `official_report:` on a
  source entry, for `stm run`) puts the Scottish Parliament's Official Report on
  the audio's clock: faster-whisper (`small`, CPU) for word timestamps only, a
  true longest-common-subsequence alignment (rapidfuzz), cues of two lines of 42
  characters and one to six seconds, the speaker named on each change. The
  Report's words are published unchanged and the track carries no `generatedBy`;
  a `NOTE` names where the words and the timing each came from, and
  `analysis.json` records the alignment summary including `unheard` words. On
  the 34-minute Party Leaders session: 4,854 words, 88.7 % anchored, 488 cues.
- **Added** Amazon Transcribe as a second clock: `--asr transcribe` on `stm
  run` and `stm captions align` (`STM_ASR=transcribe` on Batch). 83.8 % of the
  Backbench session's Report anchored against Whisper's 85.3 %; Whisper stays
  the default.
- **Added** `stm captions transcribe` for content with no written record: the
  recogniser's own words, published as machine text (`generatedBy`, `reviewed:
  false`).
- **Added** S3 input and output for running on AWS Batch: `stm run
  s3://bucket/sources.yaml`, a `media:` field on source entries (the file to
  fetch, while `url` stays the publisher's page), and upload of each finished
  title with `--upload` / `--public-base` or `STM_ASSETS_BUCKET` /
  `STM_PUBLIC_BASE`. `stm_pipeline.publish`.
- **Added** the container image build, `.github/workflows/container.yml`, over
  GitHub's OIDC token.
- **Added** `examples/scottish-parliament-fmqs-2026-09-17.json`, a real published
  title: a hand-measured side-panel crop, a 2,048-second activity track, a
  verbatim track and a simplified one marked `reviewed: false`.
- New dependency: `rapidfuzz` (core). New optional: `faster-whisper` (`captions`
  extra). Both MIT; see `DEPENDENCIES.md`.

## 0.2.0 — 2026-09-22

Format 0.2.

- **Added** `signLanguage[].activity`: the interpreter's hand and face motion
  per interval, integers 0..100 normalised to the title's own 95th percentile.
  Optional. Measured on the final crop rectangle, so hand-measured and detected
  crops both carry it. The player decides what "idle" means.
- Readers accept `stmVersion` `0.1` and `0.2`. A `0.1` manifest carrying
  `activity` is rejected, because a 0.1 reader would reject it.
- `stm activity CLIP --rect x,y,w,h` measures a rectangle on its own;
  `stm spike` prints an activity sparkline; `analysis.json` records the p95,
  sample count and idle fraction behind the published track.
- `PipelineConfig.activity`, `activity_interval_ms`, `activity_percentile`.

## 0.1.0 — 2026-09-21

First public release: the format, two JSON Schemas, `validate-stm`, the
extraction pipeline, `stm triage` and `stm spike`, and the validation harness.
