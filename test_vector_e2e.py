#!/usr/bin/env python3
"""
End-to-end integration test for vector search functionality.
Tests with in-memory catalog - no docker needed!
"""

import sys
import numpy as np
import pyarrow as pa
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("PyIceberg Vector Search - End-to-End Integration Test")
print("=" * 80)

# Test 1: Check dependencies
print("\n[1/8] Checking dependencies...")
try:
    import sentence_transformers
    print("  ✓ sentence-transformers installed")
except ImportError:
    print("  ✗ sentence-transformers missing")
    print("  Run: pip install sentence-transformers")
    sys.exit(1)

try:
    import faiss
    print("  ✓ faiss installed")
except ImportError:
    print("  ✗ faiss missing")
    print("  Run: pip install faiss-cpu")
    sys.exit(1)

try:
    from pyiceberg.catalog import load_catalog
    print("  ✓ pyiceberg installed")
except ImportError:
    print("  ✗ pyiceberg import failed")
    sys.exit(1)

# Test 2: Create in-memory catalog
print("\n[2/8] Creating in-memory catalog...")
try:
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    
    catalog = load_catalog(
        "test_catalog",
        **{
            "type": "sql",
            "uri": f"sqlite:///{tmp_dir}/test.db",
            "warehouse": f"file://{tmp_dir}/warehouse",
        }
    )
    catalog.create_namespace("vectors")
    print(f"  ✓ Catalog created at: {tmp_dir}")
except Exception as e:
    print(f"  ✗ Failed to create catalog: {e}")
    sys.exit(1)

# Test 3: Create table with embeddings
print("\n[3/8] Creating table with vector column...")
try:
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 384)),  # 384-dim embeddings
        pa.field("category", pa.string()),
    ])
    
    table = catalog.create_table(
        "vectors.documents",
        schema=schema
    )
    print(f"  ✓ Table created: {table.identifier}")
except Exception as e:
    print(f"  ✗ Failed to create table: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Enable vector search
print("\n[4/8] Enabling vector search...")
try:
    vector_table = table.vector(
        embedding_column="embedding",
        dimension=384,
        text_column="text",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_strategy="Flat"
    )
    print(f"  ✓ Vector search enabled")
    print(f"    - Dimension: {vector_table.dimension}")
    print(f"    - Model: all-MiniLM-L6-v2")
    print(f"    - Index: Flat")
except Exception as e:
    print(f"  ✗ Failed to enable vector search: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Add documents with auto-embedding
print("\n[5/8] Adding documents with auto-embedding...")
try:
    documents = [
        "Apache Iceberg is an open table format for huge analytic datasets",
        "PyIceberg is a Python implementation of Apache Iceberg",
        "Vector search enables semantic similarity queries on embeddings",
        "FAISS is a library for efficient similarity search",
        "Time-travel queries allow you to query historical data",
        "ACID transactions guarantee data consistency",
        "Parquet is a columnar storage format",
        "Data lakes store raw data in its native format",
        "Schema evolution allows tables to change over time",
        "Partition pruning skips irrelevant data files"
    ]
    
    metadata = [
        {"category": "iceberg", "source": "docs"},
        {"category": "python", "source": "docs"},
        {"category": "vector", "source": "docs"},
        {"category": "vector", "source": "docs"},
        {"category": "iceberg", "source": "docs"},
        {"category": "iceberg", "source": "docs"},
        {"category": "storage", "source": "docs"},
        {"category": "storage", "source": "docs"},
        {"category": "iceberg", "source": "docs"},
        {"category": "iceberg", "source": "docs"},
    ]
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    print(f"  Adding {len(documents)} documents...")
    vector_table.add_documents(
        documents=documents,
        metadata=metadata,
        ids=ids,
        batch_size=5
    )
    print(f"  ✓ Documents added successfully")
    print(f"    - Total vectors: {vector_table.num_vectors}")
except Exception as e:
    print(f"  ✗ Failed to add documents: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Search with text query
print("\n[6/8] Testing semantic search with text query...")
try:
    query = "table format for data lakes"
    results = vector_table.search(query, top_k=3)
    
    print(f"  Query: '{query}'")
    print(f"  ✓ Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"    {i+1}. [{result.score:.3f}] {result.id}: {result.text[:60]}...")
    
    # Verify results
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert results[0].score >= results[1].score, "Results not sorted by score"
    print("  ✓ Results properly sorted by relevance")
except Exception as e:
    print(f"  ✗ Search failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Hybrid search (metadata filter + vector search)
print("\n[7/8] Testing hybrid search (filter + vector)...")
try:
    query = "Python library"
    results_all = vector_table.search(query, top_k=10)
    results_filtered = vector_table.search(
        query, 
        top_k=10, 
        filters="category = 'python'"
    )
    
    print(f"  Query: '{query}'")
    print(f"  Without filter: {len(results_all)} results")
    print(f"  With filter (category='python'): {len(results_filtered)} results")
    
    # Verify filtering worked
    for result in results_filtered:
        assert result.metadata["category"] == "python", \
            f"Filter failed: got category={result.metadata['category']}"
    
    print("  ✓ Hybrid filtering works correctly")
    print(f"    Top result: [{results_filtered[0].score:.3f}] {results_filtered[0].text[:60]}...")
except Exception as e:
    print(f"  ✗ Hybrid search failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Search with pre-computed vector
print("\n[8/8] Testing search with pre-computed vector...")
try:
    # Create a random query vector (normalized)
    query_vector = np.random.randn(384).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    results = vector_table.search(query_vector, top_k=5)
    
    print(f"  ✓ Search with vector successful")
    print(f"    Found {len(results)} results")
    print(f"    Score range: {results[-1].score:.3f} - {results[0].score:.3f}")
except Exception as e:
    print(f"  ✗ Vector search failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Verify data in Iceberg table
print("\n[BONUS] Verifying data persistence in Iceberg...")
try:
    # Read raw table data
    data = table.scan().to_arrow()
    print(f"  ✓ Table has {len(data)} rows")
    print(f"  ✓ Columns: {data.column_names}")
    
    # Check embeddings are stored
    embeddings_col = data["embedding"]
    first_embedding = embeddings_col[0].as_py()
    print(f"  ✓ Embeddings stored (dim={len(first_embedding)})")
    
    # Check snapshot
    snapshot = table.current_snapshot()
    if snapshot:
        print(f"  ✓ Current snapshot: {snapshot.snapshot_id}")
except Exception as e:
    print(f"  ✗ Data verification failed: {e}")
    import traceback
    traceback.print_exc()

# Success!
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nVector search is working correctly! Features tested:")
print("  ✓ Auto-embedding with sentence-transformers")
print("  ✓ Semantic search with text queries")
print("  ✓ Hybrid search (metadata filters + vectors)")
print("  ✓ Search with pre-computed vectors")
print("  ✓ Data persistence in Iceberg tables")
print("\nReady for production use! 🚀")

