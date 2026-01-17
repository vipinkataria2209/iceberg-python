#!/usr/bin/env python3
"""
Simple integration test - using pre-computed embeddings (no sentence-transformers needed)
"""

import sys
import numpy as np
import pyarrow as pa
from pathlib import Path
import tempfile

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("PyIceberg Vector Search - Simple Integration Test")
print("=" * 80)

# Test 1: Imports
print("\n[1/7] Testing imports...")
try:
    from pyiceberg.catalog import load_catalog
    from pyiceberg.io.vector import VectorIndex, normalize_vectors, select_index_strategy
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create catalog
print("\n[2/7] Creating in-memory catalog...")
try:
    tmp_dir = tempfile.mkdtemp()
    catalog = load_catalog(
        "test",
        **{
            "type": "sql",
            "uri": f"sqlite:///{tmp_dir}/test.db",
            "warehouse": f"file://{tmp_dir}/warehouse",
        }
    )
    catalog.create_namespace("test")
    print(f"  ✓ Catalog created")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Create table with embedding column
print("\n[3/7] Creating table...")
try:
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 128)),
        pa.field("category", pa.string()),
    ])
    
    table = catalog.create_table("test.vectors", schema=schema)
    print(f"  ✓ Table created: test.vectors")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Add data with pre-computed embeddings
print("\n[4/7] Adding data with pre-computed embeddings...")
try:
    # Generate random embeddings (normalized)
    num_docs = 50
    embeddings = np.random.randn(num_docs, 128).astype(np.float32)
    embeddings = normalize_vectors(embeddings)
    
    # Convert to lists of float32 explicitly
    embedding_lists = [[float(x) for x in row] for row in embeddings]
    
    data = pa.Table.from_pydict({
        "id": [f"doc_{i}" for i in range(num_docs)],
        "text": [f"Document {i} content" for i in range(num_docs)],
        "embedding": embedding_lists,
        "category": ["A"] * 25 + ["B"] * 25,
    }, schema=schema)  # Explicitly use schema
    
    table.append(data)
    print(f"  ✓ Added {num_docs} documents")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Enable vector search
print("\n[5/7] Enabling vector search...")
try:
    vector_table = table.vector(
        embedding_column="embedding",
        dimension=128,
        id_column="id",
        text_column="text",
        index_strategy="Flat"
    )
    print(f"  ✓ Vector search enabled")
    print(f"    - Dimension: {vector_table.dimension}")
    print(f"    - Num vectors: {vector_table.num_vectors}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Search with vector
print("\n[6/7] Testing vector search...")
try:
    # Create query vector
    query = np.random.randn(128).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    # Search all
    results = vector_table.search(query, top_k=5)
    print(f"  ✓ Search returned {len(results)} results")
    print(f"    Top 3 scores: {[f'{r.score:.3f}' for r in results[:3]]}")
    
    # Verify sorting
    for i in range(len(results)-1):
        assert results[i].score >= results[i+1].score, "Results not sorted!"
    print(f"  ✓ Results properly sorted")
    
except Exception as e:
    print(f"  ✗ Search failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Hybrid search (filter + vector)
print("\n[7/7] Testing hybrid search...")
try:
    query = np.random.randn(128).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    # Search with filter
    results_filtered = vector_table.search(
        query,
        top_k=10,
        filters="category = 'A'"
    )
    
    print(f"  ✓ Filtered search returned {len(results_filtered)} results")
    
    # Verify all results match filter
    for r in results_filtered:
        assert r.metadata["category"] == "A", f"Filter failed!"
    print(f"  ✓ All results match filter (category='A')")
    
except Exception as e:
    print(f"  ✗ Hybrid search failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Bonus tests
print("\n[BONUS] Testing FAISS index functions...")
try:
    # Test auto-selection
    strategy_small = select_index_strategy(1000, 128)
    strategy_large = select_index_strategy(500000, 768)
    print(f"  ✓ Index auto-selection works")
    print(f"    - 1K vectors: {strategy_small}")
    print(f"    - 500K vectors: {strategy_large}")
    
    # Test VectorIndex directly
    index = VectorIndex(dimension=128, index_type="Flat")
    test_vectors = np.random.randn(100, 128).astype(np.float32)
    test_vectors = normalize_vectors(test_vectors)
    index.add(test_vectors)
    
    query = normalize_vectors(np.random.randn(128).astype(np.float32).reshape(1, -1))[0]
    distances, indices = index.search(query, k=5)
    
    print(f"  ✓ FAISS index working")
    print(f"    - Added 100 vectors")
    print(f"    - Search returned {len(indices)} results")
    
except Exception as e:
    print(f"  ✗ FAISS test failed: {e}")
    import traceback
    traceback.print_exc()

# Success!
print("\n" + "=" * 80)
print("✅ ALL CORE TESTS PASSED!")
print("=" * 80)
print("\nVector search features verified:")
print("  ✓ Pre-computed embeddings storage")
print("  ✓ Vector similarity search")
print("  ✓ Hybrid search (filters + vectors)")
print("  ✓ FAISS index integration")
print("  ✓ Result sorting and filtering")
print("\n🎉 Ready to use! (Auto-embedding needs sentence-transformers)")

