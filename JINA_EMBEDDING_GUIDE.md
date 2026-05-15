# 完整向量化和索引系统

基于KohakuRAG1的实现，本系统提供**本地Jina V4向量化 + Milvus + PostgreSQL**的完整RAG解决方案。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                  Input Documents (Text/PDF)                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
     ┌──────────────────────────┐
     │   DocumentParser         │
     │  (text_utils.py)         │
     │  + split_sentences       │
     │  + split_paragraphs      │
     └────────┬─────────────────┘
              │
              ▼
     ┌──────────────────────────┐
     │ Hierarchical Structure   │
     │  Doc → Section →         │
     │  Paragraph → Sentence    │
     └────────┬─────────────────┘
              │
              ▼
   ┌────────────────────────────────┐
   │  Jina V4 Embedder             │
   │  (jina_embedder.py)           │
   │  - Local GPU inference         │
   │  - Matryoshka dimensions       │
   │  - FP16 acceleration           │
   └────────┬──────────────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌─────────────┐ ┌──────────────┐
│   Milvus    │ │ PostgreSQL   │
│  (Vectors)  │ │  (Metadata)  │
├─────────────┤ ├──────────────┤
│ - Sentence  │ │ - Hierarchy  │
│ - Paragraph │ │ - Full text  │
│ - Section   │ │ - Metadata   │
│ - Document  │ │ - JSONB cols │
└─────────────┘ └──────────────┘
     │             │
     └─────┬───────┘
           ▼
   ┌──────────────────────┐
   │  MilvusPostgres      │
   │  NodeStore           │
   │ (unified interface)  │
   └──────────────────────┘
           │
           ▼
   ┌──────────────────────┐
   │  Search + Retrieval  │
   │  + get_context()     │
   │  + search_images()   │
   └──────────────────────┘
```

## 核心模块

### 1. **jina_embedder.py** - Jina V4本地向量化

特点：
- ✓ 自动检测GPU (CUDA > MPS > CPU)
- ✓ 支持Matryoshka维度 (128, 256, 512, 1024, 2048)
- ✓ FP16混合精度加速
- ✓ 异步接口 (asyncio)
- ✓ 懒加载模型 (首次调用时下载)

```python
from app.document_parser import JinaV4Embedder

# 初始化（首次会自动下载模型）
embedder = JinaV4Embedder(
    model_name="jinaai/jina-embeddings-v4",
    task="retrieval",           # retrieval, text-matching, code
    truncate_dim=1024,          # 输出维度
)

# 编码文本
embeddings = await embedder.embed_text(["Hello", "World"])
# 返回 shape: (2, 1024) numpy array

# 编码图像（可选）
embeddings = await embedder.embed_images([image_bytes])
```

**配置（.env）**：
```env
JINA_MODEL_NAME=jinaai/jina-embeddings-v4
JINA_EMBEDDING_DIM=1024
JINA_EMBEDDING_TASK=retrieval
```

### 2. **indexing_pipeline.py** - 完整索引流程

工作流：
```
Document Input
    ↓
[1] Build Tree (Document → Section → Paragraph → Sentence)
    ↓
[2] Embed Sentences (批处理)
    ↓
[3] Embed Paragraphs (可选: full/both mode)
    ↓
[4] Propagate Embeddings (parent = avg(children))
    ↓
[5] Convert to StoredNode
    ↓
[6] Upsert to Milvus + PostgreSQL
```

**段落嵌入模式**：
- `"averaged"` (默认): paragraph embedding = 平均子句向量
  - 计算快，节省存储，支持快速索引
- `"full"`: paragraph embedding = 直接编码段落
  - 准确度高，但额外计算成本
- `"both"`: 两者都存（averaged为主，full存在metadata）
  - 最灵活，可在检索时选择

```python
from app.document_parser import index_and_store

nodes = await index_and_store(
    text="Your document text here...",
    embedding_model=embedder,
    datastore=datastore,
    document_id="doc-001",
    title="My Document",
    paragraph_embedding_mode="averaged",  # averaged, full, both
)
```

### 3. **datastore_milvus_pg.py** - 统一存储接口

**向量存储（Milvus）**：
```
rag_nodes_vec (主索引)
├─ node_id (PK, VARCHAR)
├─ kind (VARCHAR: document/section/paragraph/sentence)
└─ embedding (FLOAT_VECTOR[1024])

rag_nodes_para_full_vec (可选, full/both模式)
└─ 完整段落向量 (较大)

rag_nodes_images_vec (可选)
└─ 图像向量
```

**元数据存储（PostgreSQL）**：
```
rag_nodes_nodes
├─ node_id (PK)
├─ parent_id (FK, 层级)
├─ kind (indexed)
├─ title
├─ text
├─ metadata (JSONB, 灵活扩展)
├─ child_ids (JSONB array)
└─ Indexes: parent_id, kind
```

**核心API**：
```python
# 存储节点
await datastore.upsert_nodes(nodes)

# 检索单个节点
node = await datastore.get_node("doc_001:sec1:p1")

# 相似度搜索
matches = await datastore.search(
    query_vector,      # 1D array, shape (1024,)
    k=5,
    kinds={NodeKind.PARAGRAPH}  # 可选过滤
)

# 获取上下文（层级遍历）
context = await datastore.get_context(
    node_id="doc_001:sec1:p1:s1",
    parent_depth=2,     # 向上 2 层
    child_depth=1,      # 向下 1 层
)

# 图像搜索（如果已索引）
image_matches = await datastore.search_images(query_vector, k=3)
```

## 完整示例

### 快速开始

```python
import asyncio
from app.document_parser import (
    JinaV4Embedder,
    MilvusPostgresNodeStore,
    index_and_store,
)

async def main():
    # 初始化
    embedder = JinaV4Embedder(truncate_dim=1024)
    datastore = MilvusPostgresNodeStore(
        dimensions=1024,
        table_prefix="my_docs",
    )
    
    # 索引文档
    nodes = await index_and_store(
        text="Your document content...",
        embedding_model=embedder,
        datastore=datastore,
        title="My Document",
    )
    
    # 搜索
    query_vec = await embedder.embed_text(["search query"])
    results = await datastore.search(query_vec[0], k=5)
    
    for match in results:
        print(f"Score: {match.score:.4f}")
        print(f"Text: {match.node.text}")

asyncio.run(main())
```

### 参考KohakuRAG1的实现细节

#### 1. 句子分割 (text_utils.py)

KohakuRAG1 vs 我们的实现：
```python
# KohakuRAG1: 英文优先，但不支持中文标点
SENTENCE_RE = re.compile(r'(?<=[.!?;])\s+')

# 我们的实现: 支持中文标点（。！？；）
SENTENCE_RE = re.compile(
    r'(?<=[.!?;。！？；])\s*'
)
```

#### 2. 段落分割 (text_utils.py)

```python
# 两层策略:
# 1. 尝试空行分割（markdown-like）
# 2. 如果只有 1 个结果，则按换行符分割

def split_paragraphs(text: str) -> list[str]:
    # 尝试空行分割
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        return [p.strip() for p in paragraphs if p.strip()]
    
    # 回退到换行符分割
    paragraphs = text.split('\n')
    return [p.strip() for p in paragraphs if p.strip()]
```

#### 3. 嵌入传播 (indexing_pipeline.py)

基于KohakuRAG1的`average_embeddings`：
```python
def average_embeddings(child_vectors: Sequence[np.ndarray]) -> np.ndarray:
    """计算子向量的归一化平均值用于父节点"""
    stacked = np.vstack(child_vectors)
    return _normalize(np.mean(stacked, axis=0, keepdims=True))[0]
```

#### 4. 树形传播 (indexing_pipeline.py)

```python
def _propagate_embeddings(node, para_full_map):
    """递归计算父节点嵌入"""
    if node.embedding is not None:
        return node.embedding  # 叶子节点
    
    # 递归获取子向量
    child_vectors = [
        self._propagate_embeddings(child, para_full_map)
        for child in node.children
    ]
    
    # 计算平均值
    averaged = average_embeddings(child_vectors)
    
    # 处理段落嵌入模式
    if node.kind == PARAGRAPH and node.id in para_full_map:
        if mode == "full":
            node.embedding = para_full_map[node.id]
        elif mode == "both":
            node.embedding = averaged
            node.metadata["full_embedding"] = para_full_map[node.id].tobytes().hex()
    else:
        node.embedding = averaged
    
    return node.embedding
```

## 性能考虑

### GPU内存需求

| 模式 | 批大小 | 显存(RTX 3090) | 吞吐量 |
|------|--------|---|---|
| FP32 | 32 | ~12GB | 2000 tokens/s |
| FP16 | 128 | ~6GB | 8000 tokens/s |
| CPU | 1 | - | 50 tokens/s |

### 优化建议

1. **启用FP16**（已默认）
   ```python
   embedder = JinaV4Embedder(device="cuda")  # 自动FP16
   ```

2. **使用averaged模式**（默认）
   ```python
   paragraph_embedding_mode="averaged"  # 比full快5倍
   ```

3. **批处理**
   ```python
   # 单独编码1000个句子
   embeddings = await embedder.embed_text(all_sentences)  # 批处理
   ```

4. **Matryoshka维度选择**
   ```python
   # 更小维度 = 更快搜索 + 更小存储
   embedder = JinaV4Embedder(truncate_dim=256)  # vs 1024
   ```

## 环境配置

### .env 模板

```env
# Milvus (向量存储)
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_DB=default

# PostgreSQL (元数据存储)
PG_DSN=postgresql://user:password@localhost:5432/ragdb

# Jina V4 嵌入模型 (本地推理)
JINA_MODEL_NAME=jinaai/jina-embeddings-v4
JINA_EMBEDDING_DIM=1024
JINA_EMBEDDING_TASK=retrieval
JINA_PARAGRAPH_MODE=averaged
```

### 依赖安装

```bash
# 从 pyproject.toml / requirements.txt 安装
pip install -e .

# 或手动安装
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install pymilvus>=2.4.0
pip install psycopg[binary]>=3.1.18
pip install pillow>=10.0.0
```

## 故障排除

### 问题: 模型下载慢

**解决**: 预下载到本地
```bash
# 使用 HF CLI
huggingface-cli download jinaai/jina-embeddings-v4
```

### 问题: GPU内存不足

**解决**: 使用CPU或降低维度
```python
# 方案 1: CPU
embedder = JinaV4Embedder(device="cpu")

# 方案 2: 降低维度
embedder = JinaV4Embedder(truncate_dim=256)

# 方案 3: 减小batch
embeddings = await embedder.embed_text(texts[:16])  # 小批
```

### 问题: Milvus连接失败

**检查**:
```bash
# 确保Milvus运行中
docker ps | grep milvus

# 检查连接
python -c "from pymilvus import connections; connections.connect(uri='http://localhost:19530')"
```

## API参考

### JinaV4Embedder

```python
class JinaV4Embedder:
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        task: str = "retrieval",  # retrieval, text-matching, code
        truncate_dim: int = 1024,  # 128, 256, 512, 1024, 2048
        device: Any | None = None,  # auto-detect
    )
    
    @property
    def dimension(self) -> int
    
    async def embed_text(texts: Sequence[str]) -> np.ndarray
    async def embed_images(image_bytes: Sequence[bytes]) -> np.ndarray
```

### DocumentIndexer

```python
class DocumentIndexer:
    def __init__(
        self,
        embedding_model: JinaV4Embedder,
        datastore: MilvusPostgresNodeStore,
        paragraph_embedding_mode: Literal["averaged", "full", "both"] = "averaged",
    )
    
    async def index_document(document: DocumentPayload) -> list[StoredNode]
    async def index_text(
        text: str,
        document_id: str | None = None,
        title: str = "Untitled",
        metadata: dict | None = None,
    ) -> list[StoredNode]
```

### MilvusPostgresNodeStore

```python
class MilvusPostgresNodeStore:
    async def upsert_nodes(nodes: Sequence[StoredNode])
    async def get_node(node_id: str) -> StoredNode
    async def search(
        query_vector: Sequence[float],
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[RetrievalMatch]
    async def get_context(
        node_id: str,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]
```

## 许可证

基于 KohakuRAG1 的实现参考。
