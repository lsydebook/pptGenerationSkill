"""Image helpers: resolve InputBundle image refs to file paths, resize, fallback."""
from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from schemas.input import ImageAsset


def resolve_image_path(asset: ImageAsset, base_dir: Path | None = None) -> str | None:
    """Return a usable on-disk path for `asset`. Materializes base64 to a tmp file."""
    if asset.path:
        p = Path(asset.path)
        if not p.is_absolute() and base_dir is not None:
            p = (base_dir / p).resolve()
        if p.exists():
            return str(p)
    if asset.base64:
        raw = base64.b64decode(asset.base64)
        fd, tmp = tempfile.mkstemp(suffix=".png", prefix=f"img_{asset.id}_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        return tmp
    return None


def make_placeholder_image(
    out_path: str | Path,
    text: str = "image",
    size: tuple[int, int] = (1280, 720),
) -> str:
    """Create a simple gray placeholder image with `text` centered."""
    img = Image.new("RGB", size, (235, 238, 244))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle(
        [(40, 40), (size[0] - 40, size[1] - 40)],
        outline=(180, 188, 200),
        width=4,
    )
    draw.text(
        ((size[0] - tw) / 2, (size[1] - th) / 2),
        text,
        fill=(80, 90, 110),
        font=font,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return str(out)


def _trim_white_borders(img: Image.Image, threshold: int = 235) -> Image.Image:
    """Crop near-white margins; threshold is the per-channel min that's still 'white'."""
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    gray = img.convert("L")
    bbox = gray.point(lambda p: 0 if p >= threshold else 255).getbbox()
    if bbox is None:
        return img
    return img.crop(bbox)


def fit_image_for_box(
    src_path: str,
    target_width_emu: int,
    target_height_emu: int,
    *,
    trim_white: bool = True,
) -> str:
    """Trim white borders then resize to exactly fit the target box.

    Returns the path to a resized image suitable for `add_picture`.
    Strategy:
      - Open with PIL
      - Trim near-white borders
      - Resize to fill target box, preserving aspect ratio (pad with white).
      - Save next to original with `.fit.png` suffix.
    """
    if target_width_emu <= 0 or target_height_emu <= 0:
        return src_path
    try:
        img = Image.open(src_path)
    except Exception:
        return src_path

    if trim_white:
        img = _trim_white_borders(img)

    src_w, src_h = img.size
    if src_h == 0:
        return src_path

    target_w_px = int(target_width_emu * 72 / 914400)
    target_h_px = int(target_height_emu * 72 / 914400)
    if target_w_px <= 0 or target_h_px <= 0:
        return src_path

    src_ratio = src_w / src_h
    target_ratio = target_w_px / target_h_px

    if abs(src_ratio - target_ratio) > 0.05:
        if src_ratio > target_ratio:
            new_w = target_w_px
            new_h = int(new_w / src_ratio)
        else:
            new_h = target_h_px
            new_w = int(new_h * src_ratio)

        if img.mode == "RGBA":
            canvas = Image.new("RGBA", (target_w_px, target_h_px), (255, 255, 255, 255))
        else:
            canvas = Image.new("RGB", (target_w_px, target_h_px), (255, 255, 255))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        offset_x = (target_w_px - new_w) // 2
        offset_y = (target_h_px - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y), resized if img.mode == "RGBA" else None)
        img = canvas
    else:
        img = img.resize((target_w_px, target_h_px), Image.LANCZOS)

    out_path = Path(src_path).with_suffix(".fit.png")
    img.save(out_path)
    return str(out_path)


def ensure_sample_images(base_dir: Path) -> None:
    """Create the placeholder images referenced by examples/input_sample.json."""
    targets = {
        base_dir / "examples" / "images" / "results.png": "F1 curve (placeholder)",
        base_dir / "examples" / "images" / "arch.png": "X-Net architecture (placeholder)",
    }
    for path, text in targets.items():
        if not path.exists():
            make_placeholder_image(path, text)
