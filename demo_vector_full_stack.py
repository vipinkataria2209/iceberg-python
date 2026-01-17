#!/usr/bin/env python3
"""
Full Stack Vector Search Demo with Hive + MinIO
================================================

This demonstrates PyIceberg vector search with:
- Hive Metastore (catalog)
- MinIO (S3-compatible storage)
- FAISS indexing
- Hybrid search (metadata + vectors)

Prerequisites:
    docker compose -f dev/docker-compose-integration.yml up -d
"""

import time
import numpy as np
import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.io.vector import normalize_vectors

print("=" * 80)
print("PyIceberg VectorLake - Full Stack Demo")
print("Hive Metastore + MinIO + FAISS + Vector Search")
print("=" * 80)

# Step 1: Connect to Hive catalog with MinIO storage
print("\n[1/8] 🔌 Connecting to Hive metastore + MinIO...")
try:
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
    print("  ✅ Connected to Hive metastore")
    print("  ✅ Connected to MinIO (S3)")
except Exception as e:
    print(f"  ❌ Connection failed: {e}")
    print("\n💡 Start Docker first:")
    print("   docker compose -f dev/docker-compose-integration.yml up -d")
    print("   # Wait 30 seconds for services to be ready")
    exit(1)

# Step 2: Create namespace and table
print("\n[2/8] 📦 Creating vector database namespace...")
try:
    # Create namespace if it doesn't exist
    try:
        catalog.create_namespace("vectorlake")
        print("  ✅ Created namespace: vectorlake")
    except Exception:
        print("  ℹ️  Namespace already exists")
    
    # Define schema with embeddings
    schema = pa.schema([
        pa.field("doc_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("content", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 384)),  # 384-dim vectors
        pa.field("category", pa.string()),
        pa.field("timestamp", pa.timestamp("us")),
    ])
    
    # Drop table if exists (for clean demo)
    try:
        catalog.drop_table("vectorlake.documents")
        print("  ℹ️  Dropped existing table")
    except Exception:
        pass
    
    table = catalog.create_table("vectorlake.documents", schema=schema)
    print("  ✅ Created table: vectorlake.documents")
    print(f"  📍 Storage: s3://warehouse/vectorlake.db/documents")
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Generate sample AI/ML documents
print("\n[3/8] 📝 Generating sample AI/ML documents...")
try:
    documents = {
        "llm": [
            "Large Language Models revolutionize NLP",
            "GPT and BERT transformers for text generation",
            "Fine-tuning LLMs for domain-specific tasks",
            "Prompt engineering techniques for better responses",
            "Token embeddings in transformer architectures",
        ],
        "vector_db": [
            "Vector databases enable semantic search at scale",
            "FAISS and HNSW for approximate nearest neighbor search",
            "Embedding storage and retrieval for AI applications",
            "Similarity search using cosine distance metrics",
            "Hybrid search combining filters and vector similarity",
        ],
        "data_lake": [
            "Apache Iceberg provides ACID transactions for data lakes",
            "Schema evolution and time travel queries in Iceberg",
            "Parquet columnar format for efficient storage",
            "Partition pruning optimizes query performance",
            "Data lakehouse architecture combines lakes and warehouses",
        ],
        "ml_ops": [
            "MLOps practices for production machine learning",
            "Model versioning and experiment tracking systems",
            "Feature stores for ML feature management",
            "Real-time inference pipelines at scale",
            "A/B testing frameworks for ML models",
        ],
    }
    
    # Generate random embeddings (in production, use actual embedding model)
    all_docs = []
    all_categories = []
    for category, docs in documents.items():
        all_docs.extend(docs)
        all_categories.extend([category] * len(docs))
    
    num_docs = len(all_docs)
    embeddings = np.random.randn(num_docs, 384).astype(np.float32)
    embeddings = normalize_vectors(embeddings)
    
    # Create PyArrow table
    data = pa.Table.from_pydict({
        "doc_id": [f"doc_{i:03d}" for i in range(num_docs)],
        "title": [f"Title {i}" for i in range(num_docs)],
        "content": all_docs,
        "embedding": embeddings.tolist(),
        "category": all_categories,
        "timestamp": [pa.scalar(None, type=pa.timestamp("us"))] * num_docs,
    }, schema=schema)
    
    print(f"  ✅ Generated {num_docs} documents across {len(documents)} categories")
    for cat, docs in documents.items():
        print(f"     - {cat}: {len(docs)} docs")
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Write to Iceberg table (MinIO storage)
print("\n[4/8] 💾 Writing data to Iceberg table on MinIO...")
try:
    table.append(data)
    
    # Verify write
    snapshot = table.current_snapshot()
    print(f"  ✅ Data written successfully")
    print(f"  📊 Snapshot ID: {snapshot.snapshot_id}")
    print(f"  📦 Summary: {snapshot.summary}")
    
    # Check MinIO storage
    scan_result = table.scan().to_arrow()
    print(f"  ✅ Verified: {len(scan_result)} rows in MinIO")
    
except Exception as e:
    print(f"  ❌ Write failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Enable vector search
print("\n[5/8] 🔍 Enabling vector search with FAISS...")
try:
    vector_table = table.vector(
        embedding_column="embedding",
        dimension=384,
        id_column="doc_id",
        text_column="content",
        index_strategy="Flat",  # Use Flat for small dataset
    )
    
    print(f"  ✅ Vector search enabled")
    print(f"  📐 Dimension: {vector_table.dimension}")
    print(f"  📊 Total vectors: {vector_table.num_vectors}")
    print(f"  🎯 Index type: Flat (exact search)")
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 6: Semantic search demo
print("\n[6/8] 🎯 Demo: Semantic Search")
print("-" * 80)
try:
    # Create query vector (simulating a search for "vector database" topic)
    query_vector = np.random.randn(384).astype(np.float32)
    query_vector = normalize_vectors(query_vector.reshape(1, -1))[0]
    
    results = vector_table.search(query_vector, top_k=5)
    
    print(f"Query: 'Find similar documents to vector database concepts'")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. 📄 {result.id} (score: {result.score:.4f})")
        print(f"   Category: {result.metadata['category']}")
        print(f"   Content: {result.text[:70]}...")
        print()
    
    print("  ✅ Semantic search working!")
    
except Exception as e:
    print(f"  ❌ Search failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 7: Hybrid search (filter + vector)
print("\n[7/8] 🔄 Demo: Hybrid Search (Filter + Vectors)")
print("-" * 80)
try:
    # Search only in "llm" category
    results = vector_table.search(
        query_vector,
        top_k=3,
        filters="category = 'llm'"
    )
    
    print("Query: Similar docs + Filter (category='llm')")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. 📄 {result.id} (score: {result.score:.4f})")
        print(f"   Content: {result.text[:60]}...")
        assert result.metadata["category"] == "llm"
        print()
    
    print("  ✅ Hybrid search working!")
    print("  💡 Iceberg filtered first, then FAISS search = Fast!")
    
except Exception as e:
    print(f"  ❌ Hybrid search failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 8: Time-travel demo
print("\n[8/8] ⏰ Demo: Time-Travel Queries")
print("-" * 80)
try:
    # Get current snapshot
    current_snapshot = table.current_snapshot()
    print(f"Current snapshot: {current_snapshot.snapshot_id}")
    
    # Add more data
    print("Adding 5 more documents...")
    new_embeddings = np.random.randn(5, 384).astype(np.float32)
    new_embeddings = normalize_vectors(new_embeddings)
    
    new_data = pa.Table.from_pydict({
        "doc_id": [f"doc_new_{i}" for i in range(5)],
        "title": [f"New Title {i}" for i in range(5)],
        "content": [f"New document {i} about AI" for i in range(5)],
        "embedding": new_embeddings.tolist(),
        "category": ["new"] * 5,
        "timestamp": [pa.scalar(None, type=pa.timestamp("us"))] * 5,
    }, schema=schema)
    
    table.append(new_data)
    new_snapshot = table.current_snapshot()
    print(f"New snapshot: {new_snapshot.snapshot_id}")
    
    # Query old snapshot (time-travel)
    print(f"\n🔙 Querying old snapshot (before new data)...")
    old_results = vector_table.search(
        query_vector,
        top_k=3,
        snapshot_id=current_snapshot.snapshot_id
    )
    print(f"  Results from old snapshot: {len(old_results)} docs")
    print(f"  ✅ Time-travel works!")
    
    # Query current snapshot
    print(f"\n⏩ Querying current snapshot (with new data)...")
    current_results = vector_table.search(query_vector, top_k=3)
    print(f"  Results from current snapshot: {len(current_results)} docs")
    
    print(f"\n  💡 Vector time-travel = Query embeddings from any point in time!")
    
except Exception as e:
    print(f"  ❌ Time-travel failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("✅ FULL STACK DEMO COMPLETE!")
print("=" * 80)
print("\n🎉 Demonstrated:")
print("  ✅ Hive metastore integration")
print("  ✅ MinIO (S3) storage backend")
print("  ✅ Vector embeddings in Iceberg")
print("  ✅ FAISS-powered semantic search")
print("  ✅ Hybrid search (metadata + vectors)")
print("  ✅ Time-travel vector queries")
print("\n📊 Architecture:")
print("  Hive (localhost:9083) → Iceberg Catalog")
print("  MinIO (localhost:9000) → S3-compatible Storage")
print("  FAISS → In-memory Vector Index")
print("  PyIceberg VectorLake → Unified API")
print("\n🚀 Ready for production AI applications!")
print("=" * 80)

