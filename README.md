# pptGenerationSkill

研究生周报 RAG 项目。

## Run

```bash
uv run main.py
```

## 项目结构

```
src/
  rag_parsing.py          # 入库主流程
  rag_retrieval.py        # 检索主流程
  config/
    env_loader.py         # .env 加载 + get_env/get_int
    llm_config.py         # Embedding + Query Planner 模型配置
    embedding.py          # EmbeddingModel（DashScope 直连）
    indexing_config.py    # 入库配置
    retrieval_config.py   # 检索流程配置（top_k、重排等）
  param/param_zh.py       # 中文提示词
  parsing/                # 解析、建树、向量化入库
  retrieval/              # Query 扩写、混合检索、上下文扩展
  storage/
  api/
```

## 日志

启动后同时输出到控制台和 `logs/YYYY-MM-DD.log`；单条格式含时间、进程、线程、源码位置。检索流程有 `step 1/5` ~ `step 5/5` 进度，异常带堆栈。

## API

- 入库：`POST /v1/parse`
- 检索：`POST /v1/retrieve`
