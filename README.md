# pptGenerationSkill

研究生周报场景的 **层级混合 RAG 服务**：文档入库 → Dense/BM25 检索 → 可选生成答案。

> `.pptx` 等仍是输入格式。PPT 生成尚未接入。本次对齐 KohakuRAG：长度加权聚合、兄弟句扩展、RRF、生成侧 `/v1/answer`。LLM Wiki 尚未加入。

**入库变更后请重新 parse。** `.env` 已将 `RAG_VEC_COLLECTION_SUFFIX` 设为 `v2`、`PARAGRAPH_MODE=both`，旧 `v1` collection 不会自动迁移。

## 能力概览

- **入库**：异步队列；落盘解析后建四层树；超长段按句切节点；叶子 embed（超窗则分窗加权平均），父节点按文本长度加权上推；段落同时写入平均向量 + `para_full`
- **检索**：Query 扩写 → Dense + BM25 → **RRF 融合** → 可选 cross-encoder rerank → 父块返回 + 兄弟句
- **生成**：`POST /v1/answer`，上下文放在问题前；abstain 时加大 k 重试；可选 ensemble
- **缓存 / 并发**：同前（Redis 版本失效、检索优先）

## 快速启动

```bash
# 配置 .env（Milvus、Embedding/Planner、Redis 等）
uv sync
uv run main.py
```

默认 `http://0.0.0.0:8000`（可用 `HOST` / `PORT` / `RELOAD` 覆盖）。

健康检查：`GET /health`（含 Redis、入库队列、检索活跃数）。

## 流水线

### 入库

```
POST /v1/parse
  → 校验 → 落盘 (data/uploads/{job_id}) → Redis job:pending → 202
Worker:
  → MarkItDown / 文本解析
  → DOCUMENT → SECTION（标题+摘要）→ PARAGRAPH（超长按句切开）→ SENTENCE（短句合并）
  → Embedding（叶子优先；超窗分窗加权平均；父节点长度加权上推；PARAGRAPH_MODE=both 时另写 para_full）
  → Milvus upsert（Dense + BM25 sparse）
  → bump 检索缓存版本 → 清理上传目录
```

支持扩展名：`.txt`（直读），以及经 MarkItDown 的 `.pdf` / `.md` / `.docx` / `.pptx` / `.xlsx` / `.html` / `.csv` 等（见 `SUPPORTED_EXTS`）。

### 检索

```
POST /v1/retrieve
  → Redis 缓存命中？
  → LLM Query 扩写（可选；在 retrieval_slot 外执行）
  → Dense + BM25 → RRF 融合 → 可选 neural rerank
  → parent-child snippet（句子命中返回段落）+ 兄弟句
  → 写缓存并返回
```

### 生成

```
POST /v1/answer
  → 与 retrieve 相同的检索
  → 上下文在前、问题在后
  → LLM JSON 作答；is_blank 则加大 top_k 重试
  → ensemble_size>1 时投票（可忽略 blank）
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/parse` | 入库（`file` / `text` / `note`）→ **202**；同一文件已在库中 → **200** `already_indexed` |
| `GET` | `/v1/jobs/{job_id}` | 轮询入库任务状态 |
| `POST` | `/v1/retrieve` | 检索 |
| `POST` | `/v1/answer` | 检索 + 生成 |

### 入库示例

```bash
curl -s -X POST http://127.0.0.1:8000/v1/parse \
  -F "file=@./report.pdf" \
  -F "note=2026级研究生新生入学须知"

# 响应：{"job_id":"...","status":"pending","poll_url":"/v1/jobs/...","queue_position":0}
# 若文件名+内容与已入库文档相同：
# {"status":"already_indexed","already_in_rag":true,"message":"该文档已在 RAG 中","document_ids":["周报-a1b2c3d4e5f6"]}
curl -s http://127.0.0.1:8000/v1/jobs/<job_id>
```

### 检索示例

```bash
curl -s -X POST http://127.0.0.1:8000/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "本周实验进展如何？",
    "use_planner": true
  }'
```

| 字段 | 说明 |
|------|------|
| `question` | 用户问题（必填） |
| `use_planner` | 是否 LLM 扩写多 query；默认 `true` |

Dense / BM25 召回数由 `.env` 的 `RETRIEVAL_TOP_K`、`RETRIEVAL_BM25_TOP_K` 决定，接口不接收。

### 生成示例

```bash
curl -s -X POST http://127.0.0.1:8000/v1/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "本周实验进展如何？",
    "use_planner": true
  }'
```

## 评测

将 `test/eval/golden_qa.example.json` 复制为 `test/eval/golden_qa.json`，补全 30～50 条并填写 `relevant_doc_ids` 或 `relevant_node_ids`：

```bash
uv run python -m test.eval.eval_retrieval --offline
uv run python -m test.eval.eval_retrieval --base-url http://127.0.0.1:8000
```

输出 hit@k 与 MRR。

## 项目结构

```
main.py                     # 入口：uvicorn + create_app()
src/
  rag_parsing.py            # 入库编排
  rag_retrieval.py          # 检索编排
  rag_answer.py             # 生成：C→Q、retry、ensemble
  api/                      # FastAPI 路由与 HTTP schema
  parsing/                  # MarkItDown、Markdown 解析、建树、向量化
  retrieval/                # Planner、RRF、rerank、上下文扩展
  storage/                  # Milvus 向量库 / BM25
  cache/                    # Redis、JobStore、检索缓存、上传落盘
  jobs/                     # 异步入库队列
  concurrency/              # 检索优先协调
  config/                   # .env 驱动配置
  param/param_zh.py         # 中文 Planner 提示词
test/                       # 单测与 probe 脚本
data/uploads/               # 入库暂存（运行时生成）
logs/                       # 按日滚动日志
```

## 配置（`.env`）

通过 `src/config/` 读取，主要变量：

| 类别 | 变量 |
|------|------|
| Milvus | `MILVUS_URI`、`MILVUS_TOKEN`、`MILVUS_DB`、`RAG_TABLE_PREFIX`、`RAG_VEC_COLLECTION_SUFFIX` |
| Embedding | `EMBEDDING_MODEL`、`EMBEDDING_DIM`、`EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY`、`EMBEDDING_MAX_CHARS` |
| Planner | `PLANNER_BASE_URL`、`PLANNER_API_KEY`、`PLANNER_MODEL`、`PLANNER_MAX_QUERIES` |
| Redis | `REDIS_URL`、`JOB_TTL_SECONDS`、`RAG_CACHE_ENABLED`、`RAG_CACHE_TTL_SECONDS` |
| 检索 | `RETRIEVAL_TOP_K`、`RETRIEVAL_BM25_TOP_K`、`RERANK_STRATEGY`（默认 `rrf`）、`INCLUDE_SIBLINGS`、`SNIPPET_RETURN_MODE` |
| Rerank | `RERANK_ENABLED`（默认开启）、`RERANK_BASE_URL`、`RERANK_MODEL`（`jina-reranker-m0`） |
| 生成 | `ANSWER_MAX_RETRIES`、`ANSWER_ENSEMBLE_SIZE`、`ANSWER_K_DELTA` |
| 其它 | `PARAGRAPH_MODE`（默认 `both`）、`MAX_PARAGRAPH_CHARS`、`MAX_SENTENCE_CHARS`、`MIN_SENTENCE_CHARS`、`UPLOAD_DIR` |

Embedding 未单独配置时，默认复用 Planner 的网关地址与 Key。

## 日志

启动后同时输出到控制台和 `logs/YYYY-MM-DD.log`；含时间、进程、线程、源码位置。检索流程有 `step 1/5` ~ `step 5/5`，异常带堆栈。

## 技术栈

Python ≥3.10 · FastAPI · uvicorn · pymilvus · OpenAI 兼容 Embedding/LLM · redis · markitdown · langchain-openai · httpx
