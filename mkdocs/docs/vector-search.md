# Vector Search

PyIceberg supports vector embeddings and semantic search natively using Apache Iceberg tables. This enables:

- **Semantic search** over your data lake
- **Time-travel queries** on historical embeddings
- **Hybrid search** combining metadata filters and vector similarity
- **ACID guarantees** for vector data

## Quick Start

```python
from pyiceberg.catalog import load_catalog
import pyarrow as pa

# Load catalog
catalog = load_catalog("default")

# Create table with embeddings
schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 768)),
    pa.field("category", pa.string())
])

table = catalog.create_table("docs.embeddings", schema=schema)

# Enable vector search
vector_table = table.vector(
    embedding_column="embedding",
    dimension=768,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Add documents (auto-embedded)
vector_table.add_documents(
    documents=["Sony wireless headphones", "JBL Bluetooth speaker"],
    metadata=[{"category": "electronics"}, {"category": "electronics"}]
)

# Search
results = vector_table.search("affordable headphones", top_k=10)
for result in results:
    print(f"{result.id}: {result.score:.3f} - {result.text}")
```

## Installation

Vector search requires additional dependencies:

```bash
pip install "pyiceberg[vector]"
```

For GPU acceleration:
```bash
pip install "pyiceberg[vector-gpu]"
```

## Creating a Vector-Enabled Table

### Option 1: Standard PyArrow Schema

```python
import pyarrow as pa

schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 768)),  # Vector column
])

table = catalog.create_table("my_namespace.vectors", schema=schema)
```

### Option 2: Using Iceberg Schema

```python
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType, ListType, FloatType

schema = Schema(
    NestedField(1, "id", StringType(), required=True),
    NestedField(2, "text", StringType()),
    NestedField(3, "embedding", ListType(
        element_id=4,
        element_type=FloatType(),
        element_required=True
    ))
)

table = catalog.create_table("my_namespace.vectors", schema=schema)
```

## Adding Data

### Auto-Embedding from Text

```python
# Enable vector search with auto-embedding
vector_table = table.vector(
    embedding_column="embedding",
    dimension=768,
    text_column="text",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Add documents - embeddings generated automatically
vector_table.add_documents(
    documents=[
        "Apache Iceberg is a table format for huge analytic datasets",
        "PyIceberg is a Python library for Apache Iceberg",
        "Vector search enables semantic similarity queries"
    ],
    metadata=[
        {"category": "lakehouse", "source": "docs"},
        {"category": "python", "source": "docs"},
        {"category": "search", "source": "docs"}
    ],
    ids=["doc1", "doc2", "doc3"]
)
```

### Pre-Computed Embeddings

```python
import numpy as np

# Enable vector search (no embedding model needed)
vector_table = table.vector(
    embedding_column="embedding",
    dimension=768
)

# Add pre-computed vectors
embeddings = np.random.randn(100, 768).astype(np.float32)
ids = [f"doc_{i}" for i in range(100)]

vector_table.add_vectors(
    ids=ids,
    embeddings=embeddings,
    metadata=[{"batch": "1"} for _ in range(100)]
)
```

## Searching

### Basic Search

```python
# Search with text (auto-embedded)
results = vector_table.search(
    query="table format for data lakes",
    top_k=10
)

for result in results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.text}")
    print(f"Metadata: {result.metadata}")
    print()
```

### Search with Pre-Computed Vector

```python
import numpy as np

query_vector = np.random.randn(768).astype(np.float32)

results = vector_table.search(
    query=query_vector,
    top_k=10
)
```

### Hybrid Search (Metadata Filtering + Vector Similarity)

```python
# Filter THEN search - highly efficient!
results = vector_table.search(
    query="Python library",
    top_k=10,
    filters="category = 'python' AND source = 'docs'"
)
```

This leverages Iceberg's partition pruning and file skipping for 10-100x speedup!

### Time-Travel Search

Query historical knowledge states:

```python
# Get historical snapshot
old_snapshot = table.history()[0].snapshot_id

# Search as of that snapshot
results = vector_table.search(
    query="machine learning",
    top_k=10,
    snapshot_id=old_snapshot
)
```

## Embedding Models

PyIceberg supports multiple embedding models via [sentence-transformers](https://www.sbert.net/):

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose (default) |
| all-mpnet-base-v2 | 768 | Medium | Excellent | High quality |
| all-MiniLM-L12-v2 | 384 | Fast | Good | Fast retrieval |
| paraphrase-multilingual-mpnet-base-v2 | 768 | Medium | Excellent | Multilingual |

### Using Different Models

```python
vector_table = table.vector(
    embedding_column="embedding",
    dimension=768,
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
```

### OpenAI Embeddings

```python
# Custom embedding function
def embed_with_openai(texts):
    import openai
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([e.embedding for e in response.data])

# Use pre-computed embeddings
embeddings = embed_with_openai(["doc1", "doc2"])
vector_table.add_vectors(ids=["1", "2"], embeddings=embeddings)
```

## Performance Tuning

### Index Strategy

PyIceberg automatically selects the optimal FAISS index based on data size:

| Data Size | Index Type | Search Speed | Accuracy |
|-----------|------------|--------------|----------|
| < 10K | Flat | Fast | 100% |
| 10K-100K | IVF100 | 10x faster | 99% |
| 100K-1M | IVF1000 | 50x faster | 98% |
| > 1M | IVF+PQ | 100x faster | 95% |

Manual override:

```python
vector_table = table.vector(
    embedding_column="embedding",
    dimension=768,
    index_strategy="IVF1000,Flat"  # Force specific index
)
```

### Distance Metrics

```python
# Cosine similarity (default, normalized vectors)
vector_table = table.vector(..., metric="cosine")

# L2 distance
vector_table = table.vector(..., metric="l2")
```

### Partitioning for Scale

Partition your table for better performance:

```python
from pyiceberg.partitioning import PartitionSpec, PartitionField
from pyiceberg.transforms import IdentityTransform

spec = PartitionSpec(
    PartitionField(source_id=4, field_id=1000, transform=IdentityTransform(), name="category")
)

table = catalog.create_table(
    "my_namespace.vectors",
    schema=schema,
    partition_spec=spec
)
```

Now searches with `filters="category = 'X'"` only scan relevant partitions!

## API Reference

### VectorTable

```python
class VectorTable:
    def __init__(
        self,
        table: Table,
        embedding_column: str,
        dimension: int,
        id_column: str = "id",
        text_column: Optional[str] = None,
        embedding_model: Optional[str] = None,
        index_strategy: str = "auto",
        metric: str = "cosine"
    )
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> None
    
    def add_vectors(
        self,
        ids: List[str],
        embeddings: Union[List[List[float]], np.ndarray],
        metadata: Optional[List[Dict]] = None
    ) -> None
    
    def search(
        self,
        query: Union[str, List[float], np.ndarray],
        top_k: int = 10,
        filters: Optional[str] = None,
        snapshot_id: Optional[int] = None
    ) -> List[SearchResult]
    
    def rebuild_index(self) -> None
    
    @property
    def num_vectors(self) -> int
```

### SearchResult

```python
@dataclass
class SearchResult:
    id: str
    score: float  # Similarity score (0-1, higher = more similar)
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None  # Raw distance metric
```

## Examples

### RAG System

```python
# Setup
vector_table = table.vector(
    embedding_column="embedding",
    dimension=768,
    text_column="content",
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)

# Add knowledge base
documents = load_documents("knowledge_base/")
vector_table.add_documents(documents)

# Query
def rag_query(question: str) -> str:
    # Retrieve relevant docs
    results = vector_table.search(question, top_k=3)
    context = "\n\n".join([r.text for r in results])
    
    # Generate answer with LLM
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return llm.generate(prompt)

answer = rag_query("How does Apache Iceberg handle schema evolution?")
```

### Deduplication

```python
def find_duplicates(threshold=0.95):
    """Find near-duplicate documents."""
    data = table.scan().to_arrow()
    
    duplicates = []
    for i, row in enumerate(data):
        embedding = np.array(row["embedding"].as_py())
        
        # Search for similar docs
        results = vector_table.search(embedding, top_k=5)
        
        # Check for high similarity (excluding self)
        for result in results[1:]:  # Skip first (self)
            if result.score > threshold:
                duplicates.append((row["id"], result.id, result.score))
    
    return duplicates
```

### Clustering

```python
from sklearn.cluster import KMeans

# Get all embeddings
data = table.scan().to_arrow()
embeddings = np.vstack(data["embedding"].to_numpy())

# Cluster
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(embeddings)

# Add cluster labels back to table
table.update_column("cluster", clusters)
```

## Limitations

- Maximum embedding dimension: 4096 (FAISS limit)
- Vector columns must use `ListType(FloatType())` or `ListType(DoubleType())`
- Auto-embedding requires sentence-transformers package
- GPU acceleration requires faiss-gpu package

## Next Steps

- [API Documentation](api.md)
- [Performance Tuning Guide](configuration.md)
- [LangChain Integration](https://pyiceberg-vectorlake.readthedocs.io)
- [LlamaIndex Integration](https://pyiceberg-vectorlake.readthedocs.io)

