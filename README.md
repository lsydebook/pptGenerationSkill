# Weekly Report PPT Agent

基于 **LangGraph + Qwen3.6 + python-pptx** 的学术周报 PPT 生成 agent，采用 **多 Agent 架构**。

上游切片模块产出 JSON（含文本块 / 图表数据 / 图片路径），本项目将其转换为一份排版整齐的学术风格 `.pptx`。

- **多 Agent 架构**：5 子 Agent（Planner / Writer / Visual / Validator / Renderer）+ 共享 State
- **Send API 并行**：WriterAgent 内部 Map-Reduce，每页 LLM 生成并行执行
- **只需 3 次 LLM 调用**，其余全为纯代码逻辑
- 纯代码背景引擎 + 排版引擎，零额外 API 消耗
- 9 种版式 × 5 种内容风格，按 `page_type` 自动选择

> 包名 `weekly-report-ppt-agent` · Python ≥ 3.10 · MIT License

---

## 核心架构

5 子 Agent 顺序编排，通过共享 `AgentState` 流转数据：

```
上游 JSON (InputBundle)
    │
    ▼
┌─ PlannerAgent ─────────────────────────────────────┐
│  ingest [纯代码] → brief_gen [1次 LLM] → outline_plan [1次 LLM]  │
│  产出: parsed + brief + outline                                   │
└────────────────────────────────────────────────────┘
    │
    ▼
┌─ WriterAgent ────────────────────────────────────────────────────┐
│  Send API Map-Reduce: dispatch → N × worker [1次 LLM/页] → reduce │
│  每页并行生成 SlideContent，reducer 合并排序 + 回填 image_hint    │
│  产出: slide_contents                                             │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ VisualAgent ────────────────────────────────┐
│  visual_plan [规则引擎] → spec_build [纯代码]  │
│  产出: visual_decisions + slide_specs         │
└───────────────────────────────────────────────┘
    │
    ▼
┌─ ValidatorAgent ──┐
│  validate [纯代码]  │
│  产出: warnings     │
└────────────────────┘
    │
    ▼
┌─ RendererAgent ───────────────────────────────┐
│  assets [纯代码] → render [纯代码]              │
│  图表渲染 (matplotlib) + AI 生图 → .pptx       │
│  产出: rendered_path                           │
└────────────────────────────────────────────────┘

---

## 项目结构

```
周报PPT/
├── cli.py                         # CLI 入口
├── pyproject.toml                 # 包定义 (入口 weekly-ppt = "cli:main")
├── config/settings.py             # 运行时配置 (环境变量/模型选择)
├── schemas/
│   ├── input.py                   # 上游 JSON 模型 (元信息/section/block/image)
│   ├── brief.py                   # WeeklyReportBrief (LLM 提炼的简报)
│   ├── outline.py                 # WeeklyDeckOutline + OutlineItem
│   ├── content.py                 # SlideContent (LLM 输出的单页内容)
│   ├── visual.py                  # VisualDecision (版式+图片策略)
│   └── slide.py                   # WeeklySlideSpec (最终 IR)
│       └── ChartSpec / TableSpec / CalloutSpec / ImageAssetRef
├── agent/
│   ├── graph.py                   # 主图编排：5 子 Agent 顺序串接
│   ├── state.py                   # AgentState（共享状态）
│   ├── subagents/                 # 5 个子 Agent subgraph
│   │   ├── planner.py             # PlannerAgent: ingest + brief_gen + outline_plan
│   │   ├── writer.py              # WriterAgent: Send API Map-Reduce 并行
│   │   ├── visual.py              # VisualAgent: visual_plan + spec_build
│   │   ├── validator.py           # ValidatorAgent: validate
│   │   └── renderer.py            # RendererAgent: assets + render
│   └── nodes/                     # 9 个节点函数（纯逻辑，被子 Agent 复用）
│       ├── ingest.py              # JSON 校验
│       ├── brief_gen.py           # LLM 提炼简报
│       ├── outline_plan.py        # LLM 规划大纲
│       ├── slide_write.py         # 原 batch 版（保留作为对照基线）
│       ├── visual_plan.py         # 规则引擎 版式/风格/图片策略
│       ├── spec_build.py          # 组装 SlideSpec
│       ├── validate.py            # 密度校验
│       ├── assets.py              # 图表 + AI 生图
│       └── render.py              # 渲染 PPTX
├── tools/
│   ├── llm.py                     # 三模型客户端 (文本/视觉/生图)
│   ├── background_engine.py       # 纯代码背景引擎 (渐变/几何/装饰)
│   ├── layout_engine.py           # 纯代码排版引擎 (版式 + 内容风格)
│   ├── image_utils.py             # 图片工具
│   └── pptx_renderer.py           # python-pptx 渲染实现
├── examples/
│   ├── input_sample.json          # 简单测试数据
│   ├── input_real.json            # 完整周报数据
│   ├── input_eeg.json             # EEG 睡眠分期示例
│   ├── input_llm.json             # LLM 长上下文压缩示例
│   ├── generate_images.py         # 示例图片生成脚本
│   └── images/                    # 已有图片素材
└── tests/
```

---

## 版式体系（Layout）

| 版式 | 用途 | 说明 |
|------|------|------|
| `cover` | 封面 | 深蓝全屏 + 居中标题/副标题 |
| `toc` | 目录 | 编号章节列表 |
| `section` | 分隔页 | 浅灰渐变底 + accent 条 + 叠加圆装饰 |
| `content` | 正文 | 标题栏 + 自适应字号 bullets + 内容风格 |
| `dual_col` | 双列正文 | 左列 1/3/5 号要点，右列 2/4/6 号 |
| `two_col` | 左文右图 | AI 生图时自动激活 |
| `image` | 大幅配图 | 标题 + 大图 + 底部要点 |
| `chart` | 图表页 | 标题 + 全宽图表区域 |
| `thanks` | 致谢 | 深蓝全屏 + 圆形装饰 |

## 内容排版风格（ContentStyle）

根据 `page_type` 自动选择：

| page_type | 风格 | 视觉效果 |
|-----------|------|----------|
| `progress` / `research` | **cards** 卡片式 | 每条要点独立浅色卡片 + 左侧 accent 条 + 底部小菱形 |
| `results` | **grid** 网格式 | 2×2 网格 + 每格左上 accent 条 + 右上淡色方块 |
| `discussion` | **separated** 分隔式 | 要点间淡蓝分隔线 + 线中间小圆点 |
| `plan` | **numbered** 编号式 | 大号 01/02/03 + 淡蓝圆衬底 + accent 横线 |
| 其他 | **default** | 单列交替底色 bullet |

所有风格共享：自适应字号（3 条 = 24pt，6 条 = 17pt）+ 交替行背景 + 页码 + 底部 callout 框。

---

## 背景引擎

纯代码生成，零 API 消耗。按页面角色分四种：

| 角色 | 对应 layout | 背景设计 |
|------|------------|----------|
| `cover` | cover | 深蓝底色 + 4 层重叠半圆 + 底部分隔线 |
| `section` | section | 灰蓝→极浅蓝渐变(20 层) + 3 个深浅不一的叠加圆 |
| `content` | content / dual_col / chart | 白→极浅蓝竖直渐变(16 层)，无其他装饰 |
| `thanks` | thanks | 深蓝底色 + 4 层重叠半圆 + 半透明白横线 |

可通过 `python tools/background_engine.py` 独立预览纯背景效果。

---

## 三种 LLM 模型

| 模型 | 用途 | 调用时机 |
|------|------|----------|
| `qwen3.6-plus` | 文本生成 | brief / outline / slide_write |
| `qwen3-vl-plus` | 视觉决策 | 仅 `--ai-image` 时优化生图 prompt |
| `wan2.6-t2i` | AI 生图 | 仅 `--ai-image` 时生成配图 |

---

## 快速上手

```powershell
# 1) 安装依赖
pip install -e ".[dev]"

# 2) 配置 API key
copy .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY

# 3) 基础生成 (3 次 LLM, 约 3 分钟)
python cli.py -i examples/input_sample.json -o out/weekly.pptx

# 4) 自定义强调色
python cli.py -i examples/input_real.json -o out/weekly.pptx --accent "#2B5B84"

# 5) 启用 AI 生图 (额外消耗 wan2.6-t2i token)
python cli.py -i examples/input_real.json --ai-image -o out/weekly.pptx
```

安装后也可用入口命令：`weekly-ppt -i ... -o ...`

---

## CLI

```
python cli.py --help

  -i, --input      输入 JSON 文件路径
  -o, --out        输出 .pptx 路径 (默认 out/weekly.pptx)
  --accent         主题强调色 hex (如 #2B5B84)
  --ai-image       启用 AI 生图 (消耗 wan2.6-t2i token)
  --log-level      日志级别 (DEBUG/INFO/WARNING/ERROR)
```

---

## 上游 JSON 格式

```json
{
  "meta": {
    "title": "第18周科研周报",
    "author": "张三",
    "date": "2026-04-30",
    "week_index": 18,
    "lab": "XX 实验室",
    "audience": "lab_meeting"
  },
  "sections": [
    {
      "id": "sec_progress",
      "title": "本周进展",
      "kind": "progress",
      "blocks": [
        {"type": "text",    "text": "本周完成了三件事。"},
        {"type": "bullets", "items": ["进展1", "进展2", "进展3"]},
        {"type": "image_ref", "image_id": "img_001", "caption": "图1"},
        {"type": "chart", "kind": "bar", "title": "性能对比",
          "categories": ["任务A", "任务B"],
          "series": [{"name": "baseline", "values": [41.7, 38.2]}]}
      ]
    }
  ],
  "images": [
    {"id": "img_001", "path": "examples/images/arch.png",
     "caption": "架构图", "description": "详细说明"}
  ]
}
```

### section.kind 与版式 / 配图策略

| section.kind | PPT 版式偏好 | 配图策略 |
|-------------|:----------:|:------:|
| `progress` | cards 卡片式 | 自动 AI 生图（实验流程风） |
| `results` | grid 网格式 | 自动 AI 生图（数据可视化风） |
| `research` | cards 卡片式 | 自动 AI 生图（架构/方法风） |
| `discussion` | separated 分隔式 | 自动 AI 生图（概念抽象风） |
| `plan` | numbered 编号式 | 不自动配图 |

### block 类型

| type | 说明 |
|------|------|
| `text` | 正文段落 |
| `bullets` | 要点列表 |
| `image_ref` | 引用 `images[]` 中的已有图片 |
| `chart` | 内置图表数据 (bar / line / pie) |
| `formula` | LaTeX 公式 |

### 图片提供方式

- **本地路径**：`"path": "examples/images/arch.png"`
- **Base64 编码**：`"base64": "iVBORw0KGgo..."`
- **AI 自动生成**：不填 images，传 `--ai-image` 即可

---

## 测试

```powershell
pytest
```

---

## License

MIT
