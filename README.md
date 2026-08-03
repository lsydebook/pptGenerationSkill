# pptGenerationSkill

研究生周报场景的 **层级混合 RAG 检索服务**：文档入库 → 向量/关键词索引 → 问题检索与上下文片段返回。

> 当前实现是知识检索层（ingest + retrieve）。PPT / 答案生成尚未接入；`.pptx` 等仅作为可解析的输入格式。

## 能力概览

- **入库**：异步队列；上传落盘后解析、建树、Embedding，写入 Milvus/Zilliz（Dense + BM25）
- **检索**：可选 LLM Query 扩写 → Dense + BM25 混合召回 → 树去重/重排 → 父子上下文扩展
- **缓存**：Redis 缓存完整检索响应；入库成功后 bump version 全局失效
- **并发**：检索优先；入库 worker 在检索活跃时可短暂让路

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
  → DOCUMENT → SECTION → PARAGRAPH → SENTENCE 建树
  → Embedding（叶子优先，父节点由子向量平均上推）
  → Milvus upsert（Dense + BM25 sparse）
  → bump 检索缓存版本 → 清理上传目录
```

支持扩展名：`.txt`（直读），以及经 MarkItDown 的 `.pdf` / `.md` / `.docx` / `.pptx` / `.xlsx` / `.html` / `.csv` 等（见 `SUPPORTED_EXTS`）。

### 检索

```
POST /v1/retrieve
  → Redis 缓存命中？
  → LLM Query 扩写（可选；在 retrieval_slot 外执行）
  → Dense + BM25 混合检索 → 去重/重排 → 上下文 snippet
  → 写缓存并返回
```

返回 `queries`、`matches`、`snippets`，不生成最终答案。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/parse` | 入库（multipart：`file` / `text` / `note`）→ **202** + `job_id` |
| `GET` | `/v1/jobs/{job_id}` | 轮询入库任务状态 |
| `POST` | `/v1/retrieve` | 检索（JSON） |

### 入库示例

```bash
curl -s -X POST http://127.0.0.1:8000/v1/parse \
  -F "file=@./report.pdf"

# 响应：{"job_id":"...","status":"pending","poll_url":"/v1/jobs/...","queue_position":0}
curl -s http://127.0.0.1:8000/v1/jobs/<job_id>
```

### 检索示例

```bash
curl -s -X POST http://127.0.0.1:8000/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "本周实验进展如何？",
    "use_planner": true,
    "top_k": 8,
    "bm25_top_k": 4
  }'
```

| 字段 | 说明 |
|------|------|
| `question` | 用户问题（必填） |
| `top_k` | 每条 query 的 Dense 召回数；默认 `RETRIEVAL_TOP_K` |
| `bm25_top_k` | 每条 query 的 BM25 召回数；`0` 关闭；默认 `RETRIEVAL_BM25_TOP_K` |
| `use_planner` | 是否 LLM 扩写多 query；默认 `true` |

## 项目结构

```
main.py                     # 入口：uvicorn + create_app()
src/
  rag_parsing.py            # 入库编排
  rag_retrieval.py          # 检索编排
  api/                      # FastAPI 路由与 HTTP schema
  parsing/                  # MarkItDown、Markdown 解析、建树、向量化
  retrieval/                # Query Planner、混合检索、上下文扩展
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
| Embedding | `EMBEDDING_MODEL`、`EMBEDDING_DIM`、`EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY` |
| Planner | `PLANNER_BASE_URL`、`PLANNER_API_KEY`、`PLANNER_MODEL`、`PLANNER_MAX_QUERIES` |
| Redis | `REDIS_URL`、`JOB_TTL_SECONDS`、`RAG_CACHE_ENABLED`、`RAG_CACHE_TTL_SECONDS` |
| 检索 | `RETRIEVAL_TOP_K`、`RETRIEVAL_BM25_TOP_K`、`RERANK_STRATEGY`、`TOP_K_FINAL`、`PARENT_DEPTH`、`CHILD_DEPTH` |
| 队列 | `INGESTION_MAX_CONCURRENT`、`INGESTION_QUEUE_MAX_SIZE`、`INGESTION_YIELD_TO_RETRIEVAL` |
| 其它 | `PARAGRAPH_MODE`、`UPLOAD_DIR`、`MAX_FILE_SIZE_MB`、`HOST`、`PORT` |

Embedding 未单独配置时，默认复用 Planner 的网关地址与 Key。

## 日志

启动后同时输出到控制台和 `logs/YYYY-MM-DD.log`；含时间、进程、线程、源码位置。检索流程有 `step 1/5` ~ `step 5/5`，异常带堆栈。

## 技术栈

Python ≥3.10 · FastAPI · uvicorn · pymilvus · OpenAI 兼容 Embedding/LLM · redis · markitdown · langchain-openai
