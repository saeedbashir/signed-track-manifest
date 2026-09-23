# Third-party models

Only permissively licensed models are accepted. The code licence and the weights
licence are checked separately: a permissive inference library with
restrictively licensed weights is the same problem wearing a different hat.

| Model | Role | Code licence | Weights | Source |
| --- | --- | --- | --- | --- |
| **YOLOX-Tiny** (default) | Person detection, 416×416, COCO classes; only class 0 (person) is used | Apache-2.0 | Distributed as a release asset (`yolox_tiny.onnx`, release `0.1.1rc0`) of the same Apache-2.0 repository, with no separate weights licence stated. Trained on COCO 2017 (annotations CC BY 4.0). SHA-256 `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`. | https://github.com/Megvii-BaseDetection/YOLOX |
| YOLOX-Nano | Faster, less accurate alternative for quick spikes | Apache-2.0 | As above; `yolox_nano.onnx`, SHA-256 `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`. | https://github.com/Megvii-BaseDetection/YOLOX |

| **Whisper `small`** (`[captions]` extra) | Speech recognition, English, word timestamps. A clock for the Official Report; a transcript only where none exists. | MIT (`faster-whisper`, CTranslate2) | OpenAI's Whisper weights, MIT, in Systran's CTranslate2 conversion (`Systran/faster-whisper-small`, MIT), fetched by `faster-whisper` from the Hugging Face Hub on first use into its own cache. `base`/`medium` are a flag away. | https://github.com/openai/whisper · https://github.com/SYSTRAN/faster-whisper |

Detector weights are downloaded on first use into `~/.cache/stm/models` (or
`STM_MODEL_DIR`) and verified against the checksums above. Whisper weights are
fetched by `faster-whisper` into the Hugging Face cache. None are committed to
this repository.

## Rejected

| Model | Why not |
| --- | --- |
| YOLOv8 (`ultralytics`) | Package and weights are AGPL-3.0. Incompatible with an Apache-2.0 distribution. |
