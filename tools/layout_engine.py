"""Programmatic layout engine. Computes element positions without templates.

Supports 8 layouts: cover, toc, section, content, two_col, image, image_only, chart, thanks.
Each layout has unique visual identity with accent bars, page numbers, bullet markers.

Units: EMU (1 inch = 914400, 1 cm = 360000). Slide: 16:9 (12192000 x 6858000).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from schemas.slide import WeeklySlideSpec

SLIDE_W = 12192000
SLIDE_H = 6858000

MARGIN_L = 900000
MARGIN_R = 900000
MARGIN_T = 540000
MARGIN_B = 540000

CONTENT_W = SLIDE_W - MARGIN_L - MARGIN_R
CONTENT_H = SLIDE_H - MARGIN_T - MARGIN_B

COLOR_BG = "FFFFFF"
COLOR_TITLE = "1A1A2E"
COLOR_BODY = "333333"
COLOR_SUBTITLE = "666666"
COLOR_ACCENT = "2B5B84"
COLOR_ACCENT_BAR = "3A7BBF"
COLOR_LIGHT_BG = "F0F4F8"
COLOR_MUTED = "999999"
COLOR_WHITE = "FFFFFF"
COLOR_PAGE_NUM = "AABBCC"
COLOR_BULLET_MARKER = "3A7BBF"

FONT_COVER_TITLE = 40
FONT_COVER_SUBTITLE = 18
FONT_SECTION = 34
FONT_TITLE = 28
FONT_SUBTITLE = 16
FONT_BODY = 16
FONT_BULLET = 15
FONT_CAPTION = 11
FONT_THANKS = 44
FONT_THANKS_SUB = 20
FONT_TOC_ITEM = 18
FONT_TOC_NUM = 22
FONT_PAGE_NUM = 9

ACCENT_BAR_W = 200000
ACCENT_BAR_H = 480000
GAP_SM = 100000
GAP_MD = 240000
PAGE_NUM_Y = SLIDE_H - MARGIN_B + 200000
BOTTOM_LINE_Y = SLIDE_H - 300000


@dataclass
class TextBox:
    x: int
    y: int
    w: int
    h: int
    text: str
    font_size: int
    bold: bool = False
    color: str = COLOR_BODY
    alignment: str = "left"
    font_name: str = "Microsoft YaHei"
    bg_color: str | None = None


@dataclass
class ImageBox:
    x: int
    y: int
    w: int
    h: int
    image_path: str


@dataclass
class ChartArea:
    x: int
    y: int
    w: int
    h: int


@dataclass
class DecorationShape:
    shape_type: str  # "rectangle" | "oval"
    x: int
    y: int
    w: int
    h: int
    color: str


@dataclass
class SlideLayout:
    slide_width: int = SLIDE_W
    slide_height: int = SLIDE_H
    bg_color: str = COLOR_BG
    text_boxes: list[TextBox] = field(default_factory=list)
    image_boxes: list[ImageBox] = field(default_factory=list)
    chart_area: ChartArea | None = None
    decorations: list[DecorationShape] = field(default_factory=list)


class LayoutEngine:
    def __init__(self, accent_color: str = COLOR_ACCENT):
        self.accent = accent_color

    def compute(self, spec: WeeklySlideSpec) -> SlideLayout:
        dispatch = {
            "cover": self._cover,
            "toc": self._toc,
            "section": self._section,
            "content": self._content,
            "dual_col": self._dual_col,
            "two_col": self._two_col,
            "image": self._image,
            "image_only": self._image_only,
            "chart": self._chart,
            "thanks": self._thanks,
        }
        handler = dispatch.get(spec.layout, self._content)
        layout = handler(spec)
        self._add_page_number(layout, spec.page_index)
        return layout

    # ------------------------------------------------------------------
    # cover: full-color bg, centered title/subtitle
    # ------------------------------------------------------------------
    def _cover(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout(bg_color=self.accent)
        title_y = SLIDE_H // 2 - 800000
        layout.text_boxes.append(TextBox(
            x=MARGIN_L, y=title_y, w=CONTENT_W, h=1200000,
            text=spec.title, font_size=FONT_COVER_TITLE, bold=True,
            color=COLOR_WHITE, alignment="center",
        ))
        if spec.subtitle:
            layout.text_boxes.append(TextBox(
                x=MARGIN_L, y=title_y + 1000000, w=CONTENT_W, h=500000,
                text=spec.subtitle, font_size=FONT_COVER_SUBTITLE,
                color="FFFFFF99", alignment="center",
            ))
        layout.text_boxes.append(TextBox(
            x=0, y=SLIDE_H - 120000, w=SLIDE_W, h=60000,
            text="", font_size=1, color="FFFFFF33",
        ))
        return layout

    # ------------------------------------------------------------------
    # toc: numbered chapter list
    # ------------------------------------------------------------------
    def _toc(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        layout.text_boxes.append(TextBox(
            x=MARGIN_L, y=MARGIN_T + 80000, w=CONTENT_W, h=700000,
            text=spec.title, font_size=FONT_SECTION, bold=True, color=COLOR_TITLE,
        ))
        self._add_bottom_line(layout)
        y0 = MARGIN_T + 900000
        for i, bullet in enumerate(spec.bullets, 1):
            layout.text_boxes.append(TextBox(
                x=MARGIN_L, y=y0, w=600000, h=450000,
                text=f"{i:02d}", font_size=FONT_TOC_NUM, bold=True,
                color=self.accent,
            ))
            layout.text_boxes.append(TextBox(
                x=MARGIN_L + 700000, y=y0, w=CONTENT_W - 700000, h=450000,
                text=bullet, font_size=FONT_TOC_ITEM, color=COLOR_TITLE,
            ))
            y0 += 500000
        return layout

    # ------------------------------------------------------------------
    # section: light bg, accent bar, big title
    # ------------------------------------------------------------------
    def _section(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout(bg_color=COLOR_LIGHT_BG)
        layout.text_boxes.append(TextBox(
            x=MARGIN_L, y=MARGIN_T + 150000, w=ACCENT_BAR_W, h=ACCENT_BAR_H,
            text="", font_size=1, color=self.accent,
        ))
        self._add_bottom_accent_strip(layout)
        layout.text_boxes.append(TextBox(
            x=MARGIN_L + ACCENT_BAR_W + GAP_MD, y=MARGIN_T + 250000,
            w=CONTENT_W - ACCENT_BAR_W - GAP_MD, h=800000,
            text=spec.title, font_size=FONT_SECTION, bold=True, color=COLOR_TITLE,
        ))
        if spec.subtitle:
            layout.text_boxes.append(TextBox(
                x=MARGIN_L + ACCENT_BAR_W + GAP_MD, y=MARGIN_T + 1000000,
                w=CONTENT_W - ACCENT_BAR_W, h=400000,
                text=spec.subtitle, font_size=FONT_SUBTITLE, color=COLOR_SUBTITLE,
            ))
        return layout

    # ------------------------------------------------------------------
    # content: dispatches to style variant
    # ------------------------------------------------------------------
    def _content(self, spec: WeeklySlideSpec) -> SlideLayout:
        style = getattr(spec, "content_style", "default")
        dispatch = {
            "cards": self._content_cards,
            "numbered": self._content_numbered,
            "separated": self._content_separated,
            "grid": self._content_grid,
        }
        handler = dispatch.get(style, self._content_default)
        return handler(spec)

    def _content_default(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        title_h = self._add_title_bar(layout, spec.title)
        y = MARGIN_T + title_h + GAP_MD
        fsz = self._bullet_font_size(spec)
        y = self._add_bullets(layout, y, spec.bullets, font_size=fsz)
        self._add_body_and_notes(layout, spec, y)
        return layout

    # -- cards: each bullet in a light-bg card with accent bar + diamond ---
    def _content_cards(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        title_h = self._add_title_bar(layout, spec.title)
        y = MARGIN_T + title_h + GAP_MD
        y = self._render_cards(layout, MARGIN_L, y, spec.bullets, spec)
        self._add_body_and_notes(layout, spec, y)
        return layout

    def _render_cards(self, layout: SlideLayout, base_x: int, y: int, bullets: list[str], spec: WeeklySlideSpec, max_w: int | None = None) -> int:
        mw = max_w or CONTENT_W
        fsz = self._bullet_font_size(spec) - 1
        row_h = fsz * 28000
        pad = 140000
        card_x = base_x + 120000
        card_w = mw - 240000
        bar_x = base_x - 200000

        for bi, b in enumerate(bullets):
            card_bg = "F2F6FC" if bi % 2 == 0 else "F8FAFD"
            card_h = row_h + pad
            layout.text_boxes.append(TextBox(
                x=bar_x, y=y, w=80000, h=int(card_h),
                text="", font_size=1, color=self.accent,
                bg_color=self.accent,
            ))
            dz = 50000
            layout.text_boxes.append(TextBox(
                x=bar_x + 15000, y=y + int(card_h) - dz - 20000,
                w=dz, h=dz,
                text="\u25c6", font_size=8, bold=False,
                color="FFFFFFCC", alignment="center",
            ))
            layout.text_boxes.append(TextBox(
                x=card_x, y=y, w=card_w, h=int(card_h),
                text="", font_size=1, color="FFFFFF",
                bg_color=card_bg,
            ))
            layout.text_boxes.append(TextBox(
                x=card_x + pad // 2, y=y + int(card_h * 0.1),
                w=card_w - pad, h=int(card_h * 0.8),
                text=b, font_size=fsz, color=COLOR_BODY,
            ))
            y += card_h + 120000
        return y

    # -- numbered: big 01/02/03 with faint circle behind each number ---
    def _content_numbered(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        title_h = self._add_title_bar(layout, spec.title)
        y = MARGIN_T + title_h + GAP_MD
        y = self._render_numbered(layout, MARGIN_L, y, spec.bullets, spec)
        self._add_body_and_notes(layout, spec, y)
        return layout

    def _render_numbered(self, layout: SlideLayout, base_x: int, y: int, bullets: list[str], spec: WeeklySlideSpec, max_w: int | None = None) -> int:
        mw = max_w or CONTENT_W
        fsz = self._bullet_font_size(spec)
        row_h = fsz * 28000
        num_w = 700000 if max_w else 1000000

        for bi, b in enumerate(bullets, 1):
            circle_size = 480000 if max_w else 700000
            layout.text_boxes.append(TextBox(
                x=base_x + 50000, y=y - 40000,
                w=circle_size, h=circle_size,
                text="", font_size=1, color="D8E4F2",
                bg_color="D8E4F2",
            ))
            layout.text_boxes.append(TextBox(
                x=base_x + 100000, y=y + 30000, w=num_w, h=int(row_h * 0.86),
                text=f"{bi:02d}", font_size=fsz + 2, bold=True,
                color=self.accent, alignment="left",
            ))
            layout.text_boxes.append(TextBox(
                x=base_x + 200000, y=y + int(row_h * 0.86) + 20000, w=num_w + 200000, h=20000,
                text="", font_size=1, color=self.accent,
                bg_color=self.accent,
            ))
            layout.text_boxes.append(TextBox(
                x=base_x + num_w + 300000, y=y,
                w=mw - num_w - 400000, h=int(row_h * 0.86),
                text=b, font_size=fsz, color=COLOR_BODY,
            ))
            y += row_h + 60000
        return y

    # -- separated: bullets with separator lines + center dot ---
    def _content_separated(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        title_h = self._add_title_bar(layout, spec.title)
        y = MARGIN_T + title_h + GAP_MD
        y = self._render_separated(layout, MARGIN_L, y, spec.bullets, spec)
        self._add_body_and_notes(layout, spec, y)
        return layout

    def _render_separated(self, layout: SlideLayout, base_x: int, y: int, bullets: list[str], spec: WeeklySlideSpec, max_w: int | None = None) -> int:
        mw = max_w or CONTENT_W
        fsz = self._bullet_font_size(spec)
        row_h = fsz * 28000

        for bi, b in enumerate(bullets):
            bullet_x = base_x + 200000
            layout.text_boxes.append(TextBox(
                x=bullet_x - 160000, y=y, w=160000, h=int(row_h * 0.86),
                text="\u25cf", font_size=max(8, fsz - 6), bold=True,
                color=COLOR_BULLET_MARKER,
            ))
            layout.text_boxes.append(TextBox(
                x=bullet_x + 80000, y=y, w=mw - 280000, h=int(row_h * 0.86),
                text=b, font_size=fsz, color=COLOR_BODY,
            ))
            y += int(row_h * 0.86)
            if bi < len(bullets):
                sep_y = y + 60000
                sep_w = mw - 1200000
                sep_x = base_x + 600000
                layout.text_boxes.append(TextBox(
                    x=sep_x, y=sep_y, w=max(sep_w, 100000), h=15000,
                    text="", font_size=1, color="D0DAEA",
                    bg_color="D0DAEA",
                ))
                dot_s = 50000
                layout.text_boxes.append(TextBox(
                    x=sep_x + sep_w // 2 - dot_s // 2, y=sep_y - dot_s // 2 + 7500,
                    w=dot_s, h=dot_s,
                    text="\u25cf", font_size=6, bold=False,
                    color="A0B8D0", alignment="center",
                ))
                y += 120000
            else:
                y += 60000
        return y

    # -- grid: 2x2 cards with tiny accent square in top-right ---
    def _content_grid(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        title_h = self._add_title_bar(layout, spec.title)
        y = MARGIN_T + title_h + GAP_MD
        gap = GAP_MD
        col_w = (CONTENT_W - gap) // 2
        row_h = 1800000
        fsz = self._bullet_font_size(spec) - 1

        for bi, b in enumerate(spec.bullets):
            col = bi % 2
            row = bi // 2
            cx = MARGIN_L + col * (col_w + gap)
            cy = y + row * (row_h + gap)
            card_bg = "F2F6FC" if (col + row) % 2 == 0 else "F8FAFD"

            layout.text_boxes.append(TextBox(
                x=cx, y=cy, w=col_w, h=row_h,
                text="", font_size=1, color="FFFFFF",
                bg_color=card_bg,
            ))
            layout.text_boxes.append(TextBox(
                x=cx + 100000, y=cy, w=60000, h=40000,
                text="", font_size=1, color=self.accent,
                bg_color=self.accent,
            ))
            # tiny square at top-right of card
            sq = 60000
            layout.text_boxes.append(TextBox(
                x=cx + col_w - sq - 80000, y=cy + 80000,
                w=sq, h=sq,
                text="", font_size=1, color="C8D6E8",
                bg_color="C8D6E8",
            ))
            layout.text_boxes.append(TextBox(
                x=cx + 200000, y=cy + 120000,
                w=col_w - 300000, h=row_h - 240000,
                text=b, font_size=fsz, color=COLOR_BODY,
            ))
        return layout

    def _add_body_and_notes(self, layout: SlideLayout, spec: WeeklySlideSpec, y: int):
        if spec.body_text:
            layout.text_boxes.append(TextBox(
                x=MARGIN_L + 200000, y=y + GAP_SM, w=CONTENT_W - 200000, h=600000,
                text=spec.body_text, font_size=FONT_BODY, color=COLOR_BODY,
            ))
        if spec.notes:
            layout.text_boxes.append(TextBox(
                x=MARGIN_L, y=SLIDE_H - MARGIN_B - 200000,
                w=CONTENT_W, h=200000,
                text=spec.notes, font_size=FONT_CAPTION, color=COLOR_MUTED,
            ))

    # ------------------------------------------------------------------
    # dual_col: title + two-column bullets (left 1/3/... right 2/4/...)
    # ------------------------------------------------------------------
    def _dual_col(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        title_h = self._add_title_bar(layout, spec.title)
        y = MARGIN_T + title_h + GAP_MD
        col_w = (CONTENT_W - GAP_MD) // 2
        gx = MARGIN_L + 200000

        left_bullets = [b for i, b in enumerate(spec.bullets) if i % 2 == 0]
        right_bullets = [b for i, b in enumerate(spec.bullets) if i % 2 == 1]
        per_col = max(len(left_bullets), len(right_bullets))
        fsz = self._bullet_font_size(spec, bullet_count=per_col)

        y_left = y
        row_h = fsz * 28000
        for bi, b in enumerate(left_bullets):
            bg = "F5F8FB" if bi % 2 == 0 else None
            layout.text_boxes.append(TextBox(
                x=gx - 160000, y=y_left, w=160000, h=int(row_h * 0.86),
                text="\u25cf", font_size=max(8, fsz - 6), bold=True,
                color=COLOR_BULLET_MARKER,
            ))
            layout.text_boxes.append(TextBox(
                x=gx + 80000, y=y_left, w=col_w - gx + MARGIN_L - 80000, h=int(row_h * 0.86),
                text=b, font_size=fsz, color=COLOR_BODY, bg_color=bg,
            ))
            y_left += row_h

        y_right = y
        gx_r = MARGIN_L + col_w + GAP_MD + 200000
        for bi, b in enumerate(right_bullets):
            bg = "F5F8FB" if bi % 2 == 0 else None
            layout.text_boxes.append(TextBox(
                x=gx_r - 160000, y=y_right, w=160000, h=int(row_h * 0.86),
                text="\u25cf", font_size=max(8, fsz - 6), bold=True,
                color=COLOR_BULLET_MARKER,
            ))
            layout.text_boxes.append(TextBox(
                x=gx_r + 80000, y=y_right, w=col_w - 200000 - 80000, h=int(row_h * 0.86),
                text=b, font_size=fsz, color=COLOR_BODY, bg_color=bg,
            ))
            y_right += row_h

        if spec.body_text:
            y_body = max(y_left, y_right) + GAP_SM
            layout.text_boxes.append(TextBox(
                x=MARGIN_L + 200000, y=y_body, w=CONTENT_W - 200000, h=600000,
                text=spec.body_text, font_size=FONT_BODY, color=COLOR_BODY,
            ))
        return layout

    # ------------------------------------------------------------------
    # two_col: left text + right image/chart
    # ------------------------------------------------------------------
    def _two_col(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        title_h = self._add_title_bar(layout, spec.title)
        y = MARGIN_T + title_h + GAP_MD
        col_w = (CONTENT_W - GAP_MD * 2) // 2

        style = getattr(spec, "content_style", "default")
        narrow_renderers = {
            "cards": self._render_cards,
            "numbered": self._render_numbered,
            "separated": self._render_separated,
        }
        renderer = narrow_renderers.get(style)
        if renderer:
            self._render_cards(layout, MARGIN_L, y, spec.bullets, spec, max_w=col_w) if style == "cards" else \
            self._render_numbered(layout, MARGIN_L, y, spec.bullets, spec, max_w=col_w) if style == "numbered" else \
            self._render_separated(layout, MARGIN_L, y, spec.bullets, spec, max_w=col_w)
        else:
            fsz = self._bullet_font_size(spec, has_image=bool(spec.images or spec.chart))
            self._add_bullets(layout, y, spec.bullets, max_w=col_w, font_size=fsz)

        right_x = MARGIN_L + col_w + GAP_MD * 2
        if spec.images:
            img_y = MARGIN_T + title_h + GAP_MD
            img_h = SLIDE_H - img_y - MARGIN_B - 200000
            layout.image_boxes.append(ImageBox(
                x=right_x, y=img_y,
                w=col_w, h=img_h, image_path=spec.images[0].path,
            ))
            caption = spec.images[0].caption if spec.images and spec.images[0].caption else ""
            if caption:
                layout.text_boxes.append(TextBox(
                    x=right_x, y=img_y + img_h + GAP_SM,
                    w=col_w, h=200000,
                    text=caption, font_size=FONT_CAPTION, color=COLOR_SUBTITLE,
                    alignment="center",
                ))
        elif spec.chart:
            chart_y = MARGIN_T + title_h + GAP_MD
            chart_h = SLIDE_H - chart_y - MARGIN_B - 200000
            layout.chart_area = ChartArea(
                x=right_x, y=chart_y, w=col_w, h=chart_h,
            )
        return layout

    # ------------------------------------------------------------------
    # image: title + large image + bullets below
    # ------------------------------------------------------------------
    def _image(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        title_h = self._add_title_bar(layout, spec.title)
        y = MARGIN_T + title_h + GAP_MD
        remaining = SLIDE_H - y - MARGIN_B
        img_h = int(remaining * 0.55)
        gap = GAP_SM
        fsz = self._bullet_font_size(spec)

        if spec.images:
            layout.image_boxes.append(ImageBox(
                x=MARGIN_L, y=y, w=CONTENT_W, h=img_h,
                image_path=spec.images[0].path,
            ))
            y += img_h + gap

        y = self._add_bullets(layout, y, spec.bullets, max_w=CONTENT_W, font_size=fsz)
        return layout

    # ------------------------------------------------------------------
    # image_only: full-slide bleed image
    # ------------------------------------------------------------------
    def _image_only(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        if spec.images:
            layout.image_boxes.append(ImageBox(
                x=0, y=0, w=SLIDE_W, h=SLIDE_H,
                image_path=spec.images[0].path,
            ))
        return layout

    # ------------------------------------------------------------------
    # chart: title + full chart area
    # ------------------------------------------------------------------
    def _chart(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout()
        title_h = self._add_title_bar(layout, spec.title)
        y = MARGIN_T + title_h + GAP_MD
        chart_h = SLIDE_H - y - MARGIN_B
        layout.chart_area = ChartArea(x=MARGIN_L, y=y, w=CONTENT_W, h=chart_h)
        return layout

    # ------------------------------------------------------------------
    # thanks: closing slide
    # ------------------------------------------------------------------
    def _thanks(self, spec: WeeklySlideSpec) -> SlideLayout:
        layout = SlideLayout(bg_color=self.accent)
        center_y = SLIDE_H // 2
        layout.text_boxes.append(TextBox(
            x=MARGIN_L, y=center_y - 600000, w=CONTENT_W, h=800000,
            text=spec.title or "致谢", font_size=FONT_THANKS, bold=True,
            color=COLOR_WHITE, alignment="center",
        ))
        if spec.subtitle:
            layout.text_boxes.append(TextBox(
                x=MARGIN_L, y=center_y + 300000, w=CONTENT_W, h=400000,
                text=spec.subtitle, font_size=FONT_THANKS_SUB,
                color="FFFFFF99", alignment="center",
            ))
        return layout

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _bullet_font_size(self, spec: WeeklySlideSpec, bullet_count: int | None = None, has_image: bool = False) -> int:
        n = bullet_count if bullet_count is not None else len(spec.bullets)
        if n <= 0:
            return FONT_BULLET

        title_bottom = MARGIN_T + 560000 + GAP_MD
        available = SLIDE_H - title_bottom - MARGIN_B - 180000

        if spec.callout:
            available -= 320000 + 120000
        if spec.table:
            rows = len(spec.table.rows) + 1
            available -= 380000 * rows + 200000
        if has_image:
            available -= 200000

        per_slot = available / n
        scale = per_slot / 420000
        scale = min(scale, 1.6)
        scale = max(scale, 0.85)

        return max(int(FONT_BULLET * scale), 11)

    def _add_title_bar(self, layout: SlideLayout, title: str) -> int:
        layout.text_boxes.append(TextBox(
            x=MARGIN_L, y=MARGIN_T - 50000, w=ACCENT_BAR_W, h=ACCENT_BAR_H,
            text="", font_size=1, color=self.accent,
        ))
        layout.text_boxes.append(TextBox(
            x=MARGIN_L + ACCENT_BAR_W + GAP_SM, y=MARGIN_T + 30000,
            w=CONTENT_W - ACCENT_BAR_W - GAP_SM, h=500000,
            text=title, font_size=FONT_TITLE, bold=True, color=COLOR_TITLE,
        ))
        self._add_bottom_line(layout)
        return 560000

    def _add_bullets(self, layout: SlideLayout, y: int, bullets: list[str], max_w: int = CONTENT_W, font_size: int = FONT_BULLET) -> int:
        bullet_x = MARGIN_L + 200000
        row_h = font_size * 28000
        for bi, b in enumerate(bullets):
            bg = "F5F8FB" if bi % 2 == 0 else None
            layout.text_boxes.append(TextBox(
                x=bullet_x - 160000, y=y, w=160000, h=int(row_h * 0.86),
                text="\u25cf", font_size=max(8, font_size - 6), bold=True,
                color=COLOR_BULLET_MARKER, alignment="left",
            ))
            txt_w = max_w - bullet_x - 80000
            layout.text_boxes.append(TextBox(
                x=bullet_x + 80000, y=y, w=txt_w, h=int(row_h * 0.86),
                text=b, font_size=font_size, color=COLOR_BODY,
                bg_color=bg,
            ))
            y += row_h
        return y

    def _add_page_number(self, layout: SlideLayout, page_num: int):
        if page_num <= 0:
            return
        layout.text_boxes.append(TextBox(
            x=SLIDE_W - MARGIN_R - 400000, y=PAGE_NUM_Y,
            w=400000, h=200000,
            text=str(page_num), font_size=FONT_PAGE_NUM,
            color=COLOR_PAGE_NUM, alignment="right",
        ))

    def _add_bg_decor(self, layout: SlideLayout, spec: WeeklySlideSpec):
        DECO_FAINT = "E6EDF4"
        DECO_VERY_FAINT = "F0F4F8"
        DECO_FAINT_BLUE = "DCE6F0"
        DECO_COVER_LINE = "FFFFFF18"

        lt = spec.layout

        if lt in ("content", "dual_col", "chart", "image"):
            circle_x = SLIDE_W - 2400000
            circle_y = -1200000
            circle_r = 4800000
            layout.decorations.append(DecorationShape(
                shape_type="oval",
                x=circle_x, y=circle_y, w=circle_r, h=circle_r,
                color=DECO_FAINT,
            ))
            layout.decorations.append(DecorationShape(
                shape_type="rectangle",
                x=MARGIN_L, y=SLIDE_H - 180000,
                w=CONTENT_W, h=12000,
                color=DECO_VERY_FAINT,
            ))

        elif lt == "section":
            layout.decorations.append(DecorationShape(
                shape_type="oval",
                x=SLIDE_W - 3000000, y=SLIDE_H - 1800000,
                w=3600000, h=2800000,
                color=DECO_FAINT_BLUE,
            ))

        elif lt in ("cover", "thanks"):
            layout.decorations.append(DecorationShape(
                shape_type="rectangle",
                x=0, y=SLIDE_H - 180000, w=SLIDE_W, h=180000,
                color=DECO_COVER_LINE,
            ))

    def _add_bottom_line(self, layout: SlideLayout):
        layout.text_boxes.append(TextBox(
            x=MARGIN_L, y=MARGIN_T + 550000,
            w=CONTENT_W, h=20000,
            text="", font_size=1, color=self.accent,
        ))

    def _add_bottom_accent_strip(self, layout: SlideLayout):
        layout.text_boxes.append(TextBox(
            x=0, y=SLIDE_H - 80000, w=SLIDE_W, h=80000,
            text="", font_size=1, color=self.accent,
        ))


def get_engine(accent_color: str | None = None) -> LayoutEngine:
    return LayoutEngine(accent_color=accent_color or COLOR_ACCENT)
