"""visual_plan: determine layout and image strategy for each slide.

Strategy: rules-first engine gives layout variety + AI image decisions.
Image prompts are built from title + bullets + image_hint for high accuracy.
Vision LLM optionally refines prompts with full page context.
"""
from __future__ import annotations

import json
import logging

from tools.llm import get_client

from ..state import AgentState

logger = logging.getLogger(__name__)

VISUAL_SYSTEM = """你是一个学术周报 PPT 视觉设计专家。请根据完整的页面内容，生成精准的 AI 配图 prompt。
你需要真正理解这页在讲什么，然后描述一张能完美匹配内容的学术插画。"""

VISUAL_PAGES = {"research", "progress", "results", "discussion"}

WEEKLY_REPORT_STYLE = (
    "weekly academic report presentation slide illustration, "
    "clean professional academic style, blue and white color palette, "
    "subtle geometric elements, modern flat design, "
    "suitable for 16:9 widescreen slide background, "
    "NO text NO letters NO words in the image"
)


def _build_image_prompt(slide) -> str:
    """Build a precise English AI image prompt from ALL available slide context."""

    parts: list[str] = []

    title_words = slide.title.replace(" ", ", ")
    image_hint = getattr(slide, "image_hint", None)

    pt = slide.page_type
    if pt == "progress":
        parts.append(f"A professional illustration representing: {title_words}")
        parts.append("visualizing research progress and workflow milestones")
    elif pt == "research":
        parts.append(f"A scientific illustration visualizing: {title_words}")
        parts.append("abstract representation of research methodology or model architecture")
    elif pt == "results":
        parts.append(f"A data analysis illustration for: {title_words}")
        parts.append("performance metrics visualization, experimental results comparison")
    elif pt == "discussion":
        parts.append(f"A conceptual illustration about: {title_words}")
        parts.append("analytical thinking, critical evaluation, academic discourse")
    else:
        parts.append(f"An academic illustration for: {title_words}")

    bullets = slide.bullets[:3] if slide.bullets else []
    if bullets:
        key_themes = " ".join(bullets)
        key_themes_en = (
            key_themes.replace("数据", "data")
            .replace("模型", "model")
            .replace("训练", "training")
            .replace("实验", "experiment")
            .replace("网络", "network")
            .replace("分析", "analysis")
            .replace("优化", "optimization")
            .replace("架构", "architecture")
            .replace("流程", "pipeline")
            .replace("验证", "validation")
            .replace("对比", "comparison")
            .replace("性能", "performance")
            .replace("研究", "research")
            .replace("测试", "test")
            .replace("设计", "design")
            .replace("评估", "evaluation")
            .replace("检测", "detection")
            .replace("预测", "prediction")
            .replace("方法", "method")
            .replace("策略", "strategy")
            .replace("效果", "effect")
            .replace("机制", "mechanism")
            .replace("系统", "system")
        )[:300]
        parts.append(f"themes: {key_themes_en}")

    if image_hint:
        parts.append(f"visual concept: {image_hint}")

    parts.append(WEEKLY_REPORT_STYLE)

    prompt = ", ".join(parts)
    if len(prompt) > 600:
        prompt = prompt[:597] + "..."
    return prompt


def _rules_based(slide) -> dict:
    layout = "content"
    image_strategy = "none"
    image_prompt = None

    pt = slide.page_type
    bullet_count = len(slide.bullets)
    has_chart = slide.chart is not None

    if pt == "cover":
        layout = "cover"
    elif pt == "toc":
        layout = "toc"
    elif pt == "section":
        layout = "section"
    elif pt == "thanks":
        layout = "thanks"
    elif has_chart:
        layout = "chart"
        image_strategy = "chart_only"
    elif pt in VISUAL_PAGES and bullet_count >= 4:
        layout = "two_col"
        image_strategy = "ai_generate"
        image_prompt = _build_image_prompt(slide)
    elif pt in VISUAL_PAGES and bullet_count >= 3:
        layout = "dual_col"
    elif pt in VISUAL_PAGES and bullet_count >= 1:
        layout = "image"
        image_strategy = "ai_generate"
        image_prompt = _build_image_prompt(slide)
    elif bullet_count >= 6:
        layout = "two_col"
    elif bullet_count >= 3:
        layout = "dual_col"
    elif bullet_count <= 2:
        layout = "image"
        image_strategy = "ai_generate"
        image_prompt = _build_image_prompt(slide)
    else:
        layout = "content"

    return {
        "layout": layout,
        "image_strategy": image_strategy,
        "image_prompt": image_prompt,
        "existing_image_id": None,
        "color_accent": None,
        "reasoning": f"rules: {pt}, {bullet_count} bullets, chart={has_chart}",
    }


def _llm_refine_prompts(contents: list, rules_results: list[dict]) -> list[dict]:
    """Use Vision LLM to refine prompts with FULL page context for higher accuracy."""
    items = []
    for i, sc in enumerate(contents):
        if rules_results[i]["image_strategy"] != "ai_generate":
            continue
        items.append({
            "index": i,
            "page_title": sc.title,
            "page_type": sc.page_type,
            "bullets": sc.bullets,
            "body_text": sc.body_text,
            "image_hint": getattr(sc, "image_hint", None),
            "draft_prompt": rules_results[i]["image_prompt"],
        })

    if not items:
        return rules_results

    items_json = json.dumps(items, ensure_ascii=False)

    prompt = f"""你是一个学术周报 PPT 视觉设计专家。以下是每页的完整内容和草稿生图 prompt。
请你真正理解每页在讲什么，然后重写一个更精准的英文 AI 生图 prompt。

要求：
- 英文，80-120 词
- 必须紧扣该页的实际内容主题（不要泛化）
- 学术周报风格：蓝白配色、简洁专业、现代扁平设计
- 适合 16:9 幻灯片背景
- 图片中不要包含任何文字、字母、标题
- 如果是模型架构类内容→抽象神经网络风格
- 如果是数据分析类→信息图/数据可视化风格
- 如果是实验进展类→实验流程/里程碑风格
- 如果是理论讨论类→概念抽象/思维导图风格

输出 JSON 数组：
[{{"index": 序号, "image_prompt": "精准的英文生图prompt"}}, ...]

页面内容：
{items_json}
"""
    client = get_client()
    try:
        data = client.vision.analyze_json(prompt, system=VISUAL_SYSTEM, max_tokens=2500)
        if not isinstance(data, list):
            data = [data]
        ref = {r.get("index", -1): r for r in data}
        for item in items:
            i = item["index"]
            if i in ref and ref[i].get("image_prompt"):
                rules_results[i]["image_prompt"] = ref[i]["image_prompt"]
                rules_results[i]["reasoning"] += " + llm_refined"
        logger.info("vision LLM refined %d prompts", len(ref))
    except Exception as e:
        logger.warning("vision LLM refinement failed: %s", e)

    return rules_results


def visual_plan(state: AgentState) -> dict:
    contents = state.get("slide_contents") or []
    accent = state.get("accent_color", "")
    enable_ai = state.get("enable_ai_image", False)

    if not contents:
        return {"visual_decisions": []}

    rules_results = [_rules_based(sc) for sc in contents]

    if not enable_ai:
        _apply_cached_image_fallback(rules_results)
    else:
        rules_results = _llm_refine_prompts(contents, rules_results)

    from schemas.visual import VisualDecision
    decisions = [VisualDecision(**r) for r in rules_results]
    _assign_content_styles(decisions)
    for d in decisions:
        logger.info("  slide: layout=%s, image=%s, style=%s", d.layout, d.image_strategy, d.content_style)
    return {"visual_decisions": decisions}


_STYLES = ["cards", "numbered", "separated", "grid"]

_STYLE_BY_PAGE_TYPE = {
    "progress": "cards",
    "research": "cards",
    "results": "grid",
    "discussion": "separated",
    "plan": "numbered",
}


def _apply_cached_image_fallback(results: list[dict]):
    from pathlib import Path
    cache_dir = Path("out/ai_images")
    for i, r in enumerate(results):
        page = i + 1
        cached = cache_dir / f"ai_{page:02d}.png"
        if cached.exists():
            r["image_strategy"] = "use_existing"
            r["image_prompt"] = None
            if r["layout"] in ("content", "dual_col"):
                r["layout"] = "two_col"
            r["reasoning"] += f" (cached ai_{page:02d}.png, layout={r['layout']})"
            continue
        if r["image_strategy"] == "ai_generate":
            r["image_strategy"] = "none"
            r["image_prompt"] = None
            r["reasoning"] += " (no cached image, no AI)"
            if r["layout"] in ("image", "image_only", "two_col"):
                r["layout"] = "content"
                r["reasoning"] += ", layout fallback to content"


def _assign_content_styles(decisions: list):
    for d in decisions:
        if d.layout in ("content", "dual_col", "two_col"):
            pt = d.reasoning.split(",")[0].replace("rules: ", "").strip() if "rules:" in d.reasoning else ""
            style = _STYLE_BY_PAGE_TYPE.get(pt, "default")
            if not style or style == "default":
                style = "default"
            d.content_style = style
            d.reasoning += f" style={d.content_style}"
            if d.layout == "dual_col":
                d.layout = "content"
