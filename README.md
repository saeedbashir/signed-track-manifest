# Signed Track Manifest (STM)

An open format for **closed signing**: sign language interpretation delivered as a
separate video layer that the viewer's player can position, resize, or hide —
plus a validator and an offline pipeline that produces it from existing
interpreted video.

Today, signed video has the interpreter burned into the picture at a position and
size chosen by whoever produced it. STM describes the same title as two layers
and a manifest, so any player that reads it can give the viewer control. The
ITU published receiver requirements for exactly this in 2013. No consumer TV
platform has shipped it.

## What this repository is, and is not

The pipeline **finds a person and cuts a rectangle.** It locates the interpreter
composited into a video, extracts that region as its own track, patches the hole
it leaves, and writes a manifest saying where the track came from and how sure
it is.

It does **not** recognise signs, interpret meaning, assess interpretation
quality, or generate anything. **We never synthesise sign language**, and the
format is built so that nobody else can quietly pass off synthesised signing as
human: every sign track carries a required `provenance` field with no default.

Machine-generated *captions* are allowed, and are always written with
`reviewed: false`. Only an explicit step naming a human reviewer can change that.

## Install

Requires Python 3.11+ and `ffmpeg`/`ffprobe` on `PATH`.

With plain pip, from a fresh clone:

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

With [uv](https://docs.astral.sh/uv/), which is how the lockfile is maintained:

```sh
uv sync --all-extras
```

Either way you get two commands: `validate-stm` and `stm`. `requirements.txt` is exported from
the uv lockfile and pins exactly what CI ran, for anyone who wants that with pip.

## Quickstart

```sh
# validate a manifest
validate-stm examples/corner-inset.json

# fetch the detector weights once (~20 MB, checksum verified)
stm models fetch

# rank candidate clips by how good a demo they would make
stm triage candidates/*.mp4 --sources

# see what the pipeline would do to one clip, and watch the result
stm spike path/to/briefing.mp4 --detections-video
open path/to/briefing.spike.mp4          # the chosen crop drawn on every frame
open path/to/briefing.fill.interpolated-surround.mp4   # the main picture with the hole patched

# process titles listed in sources.yaml into out/<id>/ with a manifest each
stm run catalogue/sources.yaml --out out --work work

# ...and generate a simplified caption track alongside the verbatim one
stm run catalogue/sources.yaml --out out --simplify
```

`stm spike` prints the cluster table — every detected person track, with its
persistence, stability, size, motion ratio and score — so you can see why it
chose what it chose.

## Choosing titles

`stm triage` ranks candidates by how good a demo they will make. It runs
detection only, about a second per clip, and reports the layout, the extraction
confidence, and a **fill difficulty** score.

Fill difficulty is the number that matters most. A corner inset is lifted out by
interpolating the vacated rectangle from the pixels around it, so the busier and
more mobile those pixels are, the more the patch shows. The score measures both:
spatial detail decides whether the patch looks smeared in a still frame,
temporal variance decides whether it flickers between frames. A side panel
scores zero, because removing a strip needs no reconstruction at all.

```
clip                      verdict    layout            fill  conf  detail  motion
03-panel-right.mp4        excellent  side-panel/right  0.00  0.70    15.3  3592.5
02-inset-tl-calm.mp4      good       corner-inset      0.00  0.93     0.6     7.5
01-inset-br-plain.mp4     marginal   corner-inset      0.42  0.70    12.9  3124.4
```

`--sources` prints `sources.yaml` skeletons with the measurable fields filled in
and every judgement left blank. Licence, production credit and how the file was
obtained are not measured and never will be: those are a person's call.

## How the pipeline decides

1. **Sample** every 5th frame at 640px wide, streamed; nothing frame-sized is kept.
2. **Detect** people with YOLOX-Tiny on onnxruntime.
3. **Identify** the interpreter: cluster detections by position over time. The
   signer is the cluster that is positionally stable, persistent across most
   frames, small relative to the frame, and high in hand-motion energy. Motion
   carries the most weight — it is what separates an interpreter from a static
   anchor or a logo.
4. **Fix the crop**: one rectangle for the whole title, from the 5th/95th
   percentile of box edges, padded 8%, snapped to even pixels. A moving crop
   wobbles; a slightly generous fixed one does not.
5. **Classify the layout**: side panel (a strip the programme was never under —
   remove it losslessly) or corner inset (composited over the picture — patch
   the hole).
6. **Produce** the signer track (H.264, ≤540p), the main track, and a poster.
7. **Score confidence** from persistence, stability and motion. Below 0.7 the
   manifest flags the title and players should badge it.
8. **Emit and validate** the manifest.

The hole is patched with ffmpeg's `delogo` interpolation softened by a blur.
Never generative inpainting: it flickers between frames and looks worse than an
honest soft patch.

## Simplified captions

`--simplify` generates a second caption track with Claude on Amazon Bedrock.
This is not a nicety. For a viewer whose first language is a signed language,
written English is a second language, and verbatim caption speed routinely
outruns their reading rate; that research is why the signer layer and a
simplified track belong in the same player.

Three rules are enforced in code rather than in prose:

- **Timings are never invented.** The model rewrites cue text only. Cue
  boundaries, order and count come from the source file.
- **A cue the model did not return keeps its original text.** Dropping a cue
  would silently remove information from an accessibility track.
- **Output is never marked reviewed.** Generated tracks carry
  `reviewed: false`, and only an explicit step naming a human reviewer can
  change that. See `Caption.generated` and `Caption.mark_reviewed`.

Needs the `aws` extra, AWS credentials, and Bedrock model access for the chosen
model. The default is `anthropic.claude-opus-5`; `--bedrock-model` switches it,
which matters because model access is granted per model in the Bedrock console
and the criteria differ. Each cue gets a word budget derived from its own
duration, so the rewrite targets a comfortable reading rate rather than a fixed
length.

## The format

See [SPEC.md](SPEC.md). Schemas live in `src/stm/schemas/` and are what
`validate-stm` enforces. The design decisions worth knowing:

- `provenance` is required, closed (`human-interpreter` | `synthesised`), and has no default.
- `syncOffsetMs` exists because interpretation lags speech; the player shifts, the pipeline never re-encodes.
- `confidence` is public. Consumers pick their own threshold.
- `signLanguage` is an array. ASL and BSL are different languages, not variants.
- Sign languages use ISO 639-3 codes (`ase`, `bfi`); spoken languages use BCP 47.
- A caption with `generatedBy` must carry `reviewed`; `reviewed: true` requires `reviewedBy`.

## Validation harness

Tuning is done against numbers, never impressions. See
[harness/README.md](harness/README.md): clips with hand-drawn ground truth,
intersection-over-union, pass at 0.85, plus a check that the crop contains the
face — facial expression carries grammar in signed languages, and a crop that
loses it is worthless at any overlap score.

Real interpreted clips have to be found, licence-checked and labelled, so
`harness/make_corpus.py` generates a synthetic corpus with exact ground truth to
work against in the meantime. It covers the layouts the pipeline must tell
apart, including a decoy: a large still anchor beside a small signing inset,
where only motion distinguishes them.

## Sources and licences

Every title records publisher, licence, attribution, production credit and how
the file was obtained **before** anything is downloaded. The ingest step refuses
platform rips (YouTube and friends): their terms forbid downloading regardless
of the video's copyright status. Get files from the publisher, an agency
download page, or a public archive.

Detector weights and their licences are listed in
[THIRD-PARTY-MODELS.md](THIRD-PARTY-MODELS.md). Only permissive licences are
accepted; the code licence and the weights licence are checked separately.

## Development

```sh
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -q
STM_RUN_INTEGRATION=1 uv run pytest -q   # also runs the test that loads real weights
```

Dependencies and why each exists: [DEPENDENCIES.md](DEPENDENCIES.md).

## Status

Version 0.1, built during the Build, Ship, Shape: Amazon Developer Hackathon
(September–October 2026) as the open-source companion to a Fire TV closed
signing player. The format is young; expect 0.x changes. Open an issue if you
build a player or a producer against it.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
