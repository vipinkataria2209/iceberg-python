#!/usr/bin/env python3
"""
Demo: OLD vs NEW Methods
=========================

Shows that we have BOTH:
1. table.scan() - Original Iceberg (NO vectors)
2. table.vector() - NEW vector search we added
"""

import numpy as np
import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.io.vector import normalize_vectors

print("=" * 80)
print("OLD vs NEW: table.scan() vs table.vector()")
print("=" * 80)

# Setup
print("\n[Setup] Connecting to Hive + MinIO...")
catalog = load_catalog(
    "hive",
    **{
        "type": "hive",
        "uri": "thrift://localhost:9083",
        "s3.endpoint": "http://localhost:9000",
        "s3.access-key-id": "admin",
        "s3.secret-access-key": "password",
        "s3.path-style-access": "true",
    }
)

# Create table
try:
    catalog.drop_table("vectorlake.demo")
except:
    pass

schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 128)),
    pa.field("category", pa.string()),
])

table = catalog.create_table("vectorlake.demo", schema=schema)

# Insert data
embeddings = normalize_vectors(np.random.randn(20, 128).astype(np.float32))
data = pa.Table.from_pydict({
    "id": [f"doc_{i}" for i in range(20)],
    "text": [f"Document {i} content" for i in range(20)],
    "embedding": embeddings.tolist(),
    "category": ["A"] * 10 + ["B"] * 10,
}, schema=schema)

table.append(data)
print("✅ Created table with 20 rows\n")

# ============================================================================
# Method 1: OLD - table.scan() (Original Iceberg)
# ============================================================================
print("=" * 80)
print("METHOD 1: table.scan() - OLD ICEBERG METHOD (NO VECTORS)")
print("=" * 80)

print("\n📊 1. Basic scan (reads all rows):")
scan_result = table.scan().to_arrow()
print(f"   Type: {type(table.scan())}")
print(f"   Result: {type(scan_result)}")
print(f"   Rows: {len(scan_result)}")
print(f"   Columns: {scan_result.column_names}")

print("\n📊 2. Filtered scan:")
filtered = table.scan().filter("category = 'A'").to_arrow()
print(f"   Rows with category='A': {len(filtered)}")

print("\n📊 3. Column projection:")
projected = table.scan().select("id", "category").to_arrow()
print(f"   Columns: {projected.column_names}")

print("\n📊 4. Can read embeddings (but NO vector search!):")
scan_with_embeddings = table.scan().to_arrow()
first_embedding = scan_with_embeddings["embedding"][0].as_py()
print(f"   First embedding: {first_embedding[:5]}... (len={len(first_embedding)})")
print("   ⚠️  But you can't do similarity search with just scan()!")

# ============================================================================
# Method 2: NEW - table.vector() (Our new addition!)
# ============================================================================
print("\n" + "=" * 80)
print("METHOD 2: table.vector() - NEW VECTOR SEARCH METHOD!")
print("=" * 80)

print("\n🔍 1. Enable vector search:")
vector_table = table.vector(
    embedding_column="embedding",
    dimension=128,
    id_column="id",
    text_column="text",
)
print(f"   Type: {type(vector_table)}")
print(f"   Class: {vector_table.__class__.__name__}")
print(f"   Dimension: {vector_table.dimension}")

print("\n🔍 2. Vector similarity search (NEW!):")
query = normalize_vectors(np.random.randn(128).astype(np.float32).reshape(1, -1))[0]
results = vector_table.search(query, top_k=5)
print(f"   Found {len(results)} similar documents:")
for i, r in enumerate(results[:3], 1):
    print(f"     {i}. {r.id} (score: {r.score:.4f})")

print("\n🔍 3. Hybrid search (filter + vector, NEW!):")
results_filtered = vector_table.search(query, top_k=5, filters="category = 'B'")
print(f"   Found {len(results_filtered)} in category B:")
for i, r in enumerate(results_filtered[:3], 1):
    print(f"     {i}. {r.id} (score: {r.score:.4f}, cat={r.metadata['category']})")

print("\n🔍 4. Still has access to original table:")
print(f"   vector_table.table = {vector_table.table}")
print(f"   Can still use: vector_table.table.scan()")

# ============================================================================
# Side-by-Side Comparison
# ============================================================================
print("\n" + "=" * 80)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                     table.scan() (OLD)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│ ✅ Read rows from Iceberg                                               │
│ ✅ Filter by metadata (category, price, etc.)                           │
│ ✅ Column projection                                                     │
│ ✅ Time-travel                                                           │
│ ✅ Partition pruning                                                     │
│ ❌ NO vector similarity search                                           │
│ ❌ NO semantic search                                                    │
│ ❌ NO nearest neighbor search                                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     table.vector() (NEW!)                               │
├─────────────────────────────────────────────────────────────────────────┤
│ ✅ Everything from table.scan() (via .table property)                   │
│ ✅ Vector similarity search                                              │
│ ✅ Semantic search                                                       │
│ ✅ Nearest neighbor search (FAISS)                                       │
│ ✅ Hybrid search (filter + vector)                                       │
│ ✅ Time-travel vector queries                                            │
│ ✅ Auto-embedding (optional)                                             │
│ ✅ Cosine/L2 distance metrics                                            │
└─────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("✅ BOTH METHODS WORK!")
print("=" * 80)

print("\nUse Cases:")
print("  • table.scan()     → Regular data queries (no vectors needed)")
print("  • table.vector()   → Semantic/similarity search on embeddings")
print("\n💡 table.vector() is a NON-BREAKING addition!")
print("   Old code using table.scan() still works perfectly!")
print("=" * 80)

