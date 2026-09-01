"""PNG export for BenchOrStart share cards.

Uses Pillow (already pulled in by Streamlit / matplotlib). No extra deps.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

from fantasy.cards import LABEL_TONES, ShareCardView, normalize_stat_line
from fantasy.copy import EARLY_MODEL_BADGE, PRODUCT_NAME

_WIDTH = 840
_PAD = 44
_BG = (22, 27, 34)
_INK = (230, 237, 243)
_MUTED = (177, 186, 196)
_DIM = (139, 148, 158)
_WORDMARK = (248, 81, 73)
_BADGE_INK = (13, 17, 23)
_EARLY = (210, 153, 34)
_BORDER = (48, 54, 61)

_FONT_CANDIDATES = (
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ),
    (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ),
    (
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ),
)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    raw = value.lstrip("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _font_paths() -> tuple[str | None, str | None]:
    for bold_path, regular_path in _FONT_CANDIDATES:
        if Path(bold_path).is_file() and Path(regular_path).is_file():
            return bold_path, regular_path
    return None, None


_BOLD_PATH, _REGULAR_PATH = _font_paths()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = _BOLD_PATH if bold else _REGULAR_PATH
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap(text: str, width: int) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def render_share_card_png(view: ShareCardView) -> bytes:
    """Rasterize a share card. Safe to call from Streamlit download buttons."""
    tone = _hex_rgb(LABEL_TONES.get(view.label, "#58a6ff"))
    title_font = _font(36, bold=True)
    prompt_font = _font(28, bold=True)
    wordmark_font = _font(16, bold=True)
    label_font = _font(15, bold=True)
    body_font = _font(20)
    stat_font = _font(20, bold=True)
    asof_font = _font(15)

    reason_lines = _wrap(normalize_stat_line(view.reason), 52)
    stat_line = normalize_stat_line(view.stat_line)
    content_h = 220 + (28 if view.rank_line else 0)
    content_h += 48 if view.headline else 0
    content_h += 28 if view.subtitle else 0
    content_h += 32 if stat_line else 0
    content_h += 28 * max(len(reason_lines), 0)
    content_h += 36 if view.as_of_date else 0
    height = max(420, content_h + 2 * _PAD)

    image = Image.new("RGB", (_WIDTH, height), _BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, _WIDTH, 8), fill=tone)
    draw.rectangle((0, 8, 2, height), fill=_BORDER)
    draw.rectangle((_WIDTH - 2, 8, _WIDTH, height), fill=_BORDER)
    draw.rectangle((0, height - 2, _WIDTH, height), fill=_BORDER)

    x = _PAD
    y = _PAD + 4
    draw.text((x, y), PRODUCT_NAME.upper(), font=wordmark_font, fill=_WORDMARK)
    y += 28
    draw.text((x, y), view.prompt, font=prompt_font, fill=_INK)
    y += 48

    label = view.label or "START"
    bbox = draw.textbbox((0, 0), label, font=label_font)
    pill_w = bbox[2] - bbox[0] + 28
    pill_h = bbox[3] - bbox[1] + 16
    draw.rounded_rectangle((x, y, x + pill_w, y + pill_h), radius=pill_h // 2, fill=tone)
    draw.text((x + 14, y + 6), label, font=label_font, fill=_BADGE_INK)

    if view.early_model:
        badge = EARLY_MODEL_BADGE.upper()
        bb = draw.textbbox((0, 0), badge, font=label_font)
        bw = bb[2] - bb[0] + 24
        bx = x + pill_w + 12
        draw.rounded_rectangle(
            (bx, y, bx + bw, y + pill_h),
            radius=pill_h // 2,
            outline=_EARLY,
            width=2,
        )
        draw.text((bx + 12, y + 6), badge, font=label_font, fill=_EARLY)
    y += pill_h + 18

    if view.rank_line:
        draw.text((x, y), view.rank_line, font=body_font, fill=_DIM)
        y += 28
    if view.headline:
        draw.text((x, y), view.headline, font=title_font, fill=_INK)
        y += 48
    if view.subtitle:
        draw.text((x, y), view.subtitle, font=body_font, fill=_MUTED)
        y += 30
    if stat_line:
        draw.text((x, y), stat_line, font=stat_font, fill=_INK)
        y += 32
    for line in reason_lines:
        draw.text((x, y), line, font=body_font, fill=_MUTED)
        y += 28
    if view.as_of_date:
        y += 8
        draw.text((x, y), f"as of {view.as_of_date}", font=asof_font, fill=_DIM)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
