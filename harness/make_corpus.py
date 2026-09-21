#!/usr/bin/env python3
"""Generate a synthetic interpreted-broadcast corpus with known ground truth.

The validation harness needs clips whose correct answer is known. Real
interpreted content has to be found, licence-checked and hand-labelled; this
generates the layouts the pipeline must tell apart, with exact rectangles, so
the harness can be exercised before any real clip exists.

What it composites, and why each part matters:

* **A real person, cropped tightly, on a flat backdrop.** A drawn figure is not
  a person to a COCO detector -- it produces zero detections. A photograph used
  whole is no better, because the pixels around the detected person are more
  photograph, so window growth has internal structure to cross that real
  content does not have. Cropping tightly to the person and placing that on a
  flat panel is what interpreted content actually looks like.
* **Hands that move.** Drawn over the chest and oscillating, so the interpreter
  carries upper-body motion. Motion is the heaviest-weighted signal in signer
  selection, and a corpus without it tests the wrong thing.
* **A decoy anchor.** Same person, larger, hands at rest. Selecting the small
  moving figure over the large still one is the whole heuristic.

Usage:

    python harness/make_corpus.py --person PORTRAIT.jpg --out harness/corpus

`--person` is any image containing a person; the script finds and crops them.
A public-domain portrait is fine. The image is not committed to this repository.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import subprocess
import sys
from collections.abc import Callable

import cv2
import numpy as np
import numpy.typing as npt

from stm.model import Rect
from stm_pipeline.harness.ground_truth import GroundTruth, save_ground_truth

W, H, FPS, SECS = 1280, 720, 25, 6
N = FPS * SECS
BACKDROP = (38, 32, 28)
SKIN = (120, 150, 190)

Frame = npt.NDArray[np.uint8]


def crop_person(image_path: pathlib.Path) -> Frame:
    """Find the largest person in an image and return a tight crop of them."""
    from stm_pipeline.detect.onnx_person import OnnxPersonDetector
    from stm_pipeline.types import SampledFrame

    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    scale = 900 / img.shape[1]
    small = cv2.resize(img, (900, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    boxes = OnnxPersonDetector("yolox-tiny", score_threshold=0.5).detect(
        SampledFrame(0, 0.0, np.asarray(small, dtype=np.uint8), scale)
    )
    if not boxes:
        raise SystemExit(f"no person found in {image_path}")
    b = max(boxes, key=lambda x: x.area)
    pad = 0.04
    x1, y1 = max(0, int(b.x1 - b.w * pad)), max(0, int(b.y1 - b.h * pad))
    x2 = min(small.shape[1], int(b.x2 + b.w * pad))
    y2 = min(small.shape[0], int(b.y2 + b.h * pad))
    return np.asarray(small[y1:y2, x1:x2], dtype=np.uint8)


def programme(i: int, busy: bool) -> Frame:
    """The main picture. Busy carries a lot of temporal variance; calm barely any."""
    f = np.zeros((H, W, 3), np.uint8)
    if busy:
        yy, xx = np.mgrid[0:H, 0:W]
        f[..., 0] = ((xx * 2 + i * 9) % 256).astype(np.uint8)
        f[..., 1] = ((yy * 2 + i * 5) % 256).astype(np.uint8)
        f[..., 2] = (((xx + yy) + i * 13) % 256).astype(np.uint8)
        for k in range(6):
            cx = int(W / 2 + 380 * math.cos(i * 0.09 + k))
            cy = int(H / 2 + 230 * math.sin(i * 0.11 + k))
            cv2.circle(f, (cx, cy), 60, (255, 255, 255), -1)
    else:
        g = np.linspace(40, 90, W, dtype=np.float32)[None, :].repeat(H, 0) + 4 * math.sin(i * 0.05)
        f[..., 0] = np.clip(g + 20, 0, 255).astype(np.uint8)
        f[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
        f[..., 2] = np.clip(g - 10, 0, 255).astype(np.uint8)
    return f


class Corpus:
    def __init__(self, person: Frame, out: pathlib.Path) -> None:
        self.person = person
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.truth: list[GroundTruth] = []

    def sized(self, width: int) -> Frame:
        h = round(self.person.shape[0] * width / self.person.shape[1])
        return np.asarray(
            cv2.resize(self.person, (width, h), interpolation=cv2.INTER_AREA), dtype=np.uint8
        )

    @staticmethod
    def hands(img: Frame, i: int, moving: bool) -> None:
        h, w = img.shape[:2]
        r = max(4, int(w * 0.075))
        phase = i * 0.6 if moving else 0.0
        amp_x = w * 0.16 if moving else 0.0
        amp_y = h * 0.09 if moving else 0.0
        for sign in (-1, 1):
            hx = int(w / 2 + sign * (w * 0.17 + amp_x * abs(math.sin(phase + sign))))
            hy = int(h * 0.62 + amp_y * math.sin(phase * 1.3 + sign))
            cv2.circle(img, (hx, hy), r, SKIN, -1)

    def window(
        self, width: int, i: int, pad_x: int, pad_y: int, box: tuple[int, int] | None = None
    ) -> Frame:
        fig = self.sized(width).copy()
        self.hands(fig, i, True)
        ww = box[0] if box else fig.shape[1] + 2 * pad_x
        wh = box[1] if box else fig.shape[0] + 2 * pad_y
        panel = np.full((wh, ww, 3), BACKDROP, np.uint8)
        ox = (ww - fig.shape[1]) // 2
        oy = min(pad_y, max(0, wh - fig.shape[0]))
        fh = min(fig.shape[0], wh - oy)
        panel[oy : oy + fh, ox : ox + fig.shape[1]] = fig[:fh]
        return np.asarray(panel, dtype=np.uint8)

    def render(self, name: str, fn: Callable[[int], Frame], rect: Rect | None, notes: str) -> None:
        path = self.out / f"{name}.mp4"
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{W}x{H}",
                "-r",
                str(FPS),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for i in range(N):
            proc.stdin.write(np.ascontiguousarray(fn(i)).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"ffmpeg failed writing {path}")
        if rect is not None:
            self.truth.append(GroundTruth(clip=path, rect=rect, notes=notes))
        print(f"{name:24} {notes}")

    def inset(self, name: str, fig_w: int, busy: bool, corner: str, window: bool = True) -> None:
        probe = self.window(fig_w, 0, 40, 34) if window else self.sized(fig_w)
        ww, wh = probe.shape[1], probe.shape[0]
        wx = 40 if "l" in corner else W - ww - 40
        wy = 40 if "t" in corner else H - wh - 40

        def fn(i: int) -> Frame:
            f = programme(i, busy)
            patch = self.window(fig_w, i, 40, 34) if window else self.sized(fig_w).copy()
            if not window:
                self.hands(patch, i, True)
            f[wy : wy + wh, wx : wx + ww] = patch
            return f

        kind = "corner inset" + ("" if window else ", no window: signer straight on the picture")
        self.render(
            name,
            fn,
            Rect(wx, wy, ww, wh) if window else None,
            f"{kind}, {'busy' if busy else 'calm'} programme",
        )

    def panel(self, name: str, edge: str, pw: int = 300) -> None:
        px = 0 if edge == "left" else W - pw

        def fn(i: int) -> Frame:
            f = programme(i, True)
            f[:, px : px + pw] = self.window(int(pw * 0.62), i, 0, 0, box=(pw, H))
            return f

        self.render(name, fn, Rect(px, 0, pw, H), f"side panel on the {edge}")

    def anchor_and_signer(self, name: str) -> None:
        probe = self.window(190, 0, 36, 30)
        sw, sh = probe.shape[1], probe.shape[0]
        sx, sy = W - sw - 40, H - sh - 40

        def fn(i: int) -> Frame:
            f = programme(i, False)
            big = self.sized(430).copy()
            self.hands(big, i, False)
            f[60 : 60 + min(big.shape[0], H - 60), 150 : 150 + big.shape[1]] = big[: H - 60]
            f[sy : sy + sh, sx : sx + sw] = self.window(190, i, 36, 30)
            return f

        self.render(
            name,
            fn,
            Rect(sx, sy, sw, sh),
            "large still anchor plus small signing inset: motion must decide",
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--person", type=pathlib.Path, required=True, help="an image containing a person"
    )
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("harness/corpus"))
    ap.add_argument(
        "--ground-truth", type=pathlib.Path, default=pathlib.Path("harness/ground_truth.yaml")
    )
    args = ap.parse_args(argv)

    c = Corpus(crop_person(args.person), args.out)
    c.inset("01-inset-br-plain", 200, True, "br")
    c.inset("02-inset-tl-calm", 200, False, "tl")
    c.panel("03-panel-right", "right")
    c.panel("04-panel-left", "left")
    c.inset("05-inset-no-window", 200, True, "br", window=False)
    c.anchor_and_signer("06-anchor-and-signer")

    save_ground_truth(args.ground_truth, c.truth)
    print(f"\n{len(c.truth)} clips with ground truth -> {args.ground_truth}")
    print(f"score them with:  stm harness score {args.ground_truth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
