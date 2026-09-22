# Changelog

Format versions and library releases. The format version is `stmVersion` in a
manifest; the library version is `pyproject.toml`.

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
