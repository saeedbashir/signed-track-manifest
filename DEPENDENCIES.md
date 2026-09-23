# Dependencies, and why each one is here

A dependency is added only with a reason recorded here.

## Runtime

| Package | Licence | Why |
| --- | --- | --- |
| `numpy` | BSD-3 | Array maths for decode, percentiles, motion energy, running statistics. |
| `opencv-python` | Apache-2.0 | Video decoding and frame sampling (`VideoCapture.grab` skips frames without decoding), resizing, grey conversion, and the `selectROI` window used to draw ground truth. The GUI build rather than `-headless` because labelling needs a window; CI installs `libgl1`. |
| `onnxruntime` | MIT | Runs the YOLOX ONNX model on CPU. No PyTorch, no CUDA, no vendor SDK. |
| `jsonschema` | MIT | Draft 2020-12 validation, which the STM schemas use for `if`/`then` rules. |
| `pyyaml` | MIT | `sources.yaml` and `ground_truth.yaml`. |
| `rapidfuzz` | MIT | A true longest common subsequence (`distance.LCSseq`) for aligning a publisher's transcript to recognised words. The standard library's `difflib` was tried first: it is greedy on the longest block, and a chamber repeats itself, so the second copy of a phrase was pinned to the first utterance whenever one misheard word made that block longer. The optimum matches both copies; on a 34-minute session it put 50 more words where they were said. C++ with wheels, no runtime beyond itself. |
| **ffmpeg** (binary, not a package) | LGPL/GPL build-dependent | All media transforms via `subprocess`. Chosen over the `ffmpeg-python` wrapper the original spec named: one fewer dependency, and the filter graphs are written out explicitly in `ffmpeg.py` where they can be read. |

## Optional

| Package | Licence | Why |
| --- | --- | --- |
| `anthropic[bedrock]` (`[aws]` extra) | MIT | `AnthropicBedrockMantle`, the Claude client for the Bedrock Messages API endpoint, used by `--simplify`. Preferred over hand-rolling `InvokeModel` request bodies on `boto3`: SigV4 signing, retries and typed responses come for free, and the request shape matches the first-party API. |
| `boto3` (`[aws]` extra) | Apache-2.0 | S3 sync from the extraction container. |
| `faster-whisper` (`[captions]` extra) | MIT | Whisper on CTranslate2, CPU, int8 — about five times real time on a laptop with the `small` model. Used as a *clock*: a publisher's written transcript is aligned to its word timings and published unchanged, so the caption text stays human. Where no transcript exists it is the transcript, and that track is always `generatedBy`/`reviewed: false`. Chosen over `openai-whisper` (needs torch, ~2 GB) and over cloud transcription (a paid service, and the audio would leave the machine). |

## Development

| Package | Why |
| --- | --- |
| `pytest` | Tests. |
| `ruff` | Lint and format. |
| `mypy` (strict) | Types. `cv2` and `onnxruntime` have no stubs and are ignored for import. |
| `types-PyYAML`, `types-jsonschema` | Stubs. |

## Deliberately absent

- **`ultralytics` / YOLOv8** — AGPL-3.0, incompatible with shipping this repository under Apache 2.0.
- **Any generative inpainting model** — the fill is interpolation and blur, never synthesis.
- **A CLI framework** — `argparse` is enough for six subcommands.
