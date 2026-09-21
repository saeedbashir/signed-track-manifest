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
| **ffmpeg** (binary, not a package) | LGPL/GPL build-dependent | All media transforms via `subprocess`. Chosen over the `ffmpeg-python` wrapper the original spec named: one fewer dependency, and the filter graphs are written out explicitly in `ffmpeg.py` where they can be read. |

## Optional

| Package | Licence | Why |
| --- | --- | --- |
| `anthropic[bedrock]` (`[aws]` extra) | MIT | `AnthropicBedrockMantle`, the Claude client for the Bedrock Messages API endpoint, used by `--simplify`. Preferred over hand-rolling `InvokeModel` request bodies on `boto3`: SigV4 signing, retries and typed responses come for free, and the request shape matches the first-party API. |
| `boto3` (`[aws]` extra) | Apache-2.0 | S3 sync from the extraction container. |

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
