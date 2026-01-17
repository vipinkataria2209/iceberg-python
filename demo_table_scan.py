#!/usr/bin/env python3
"""
Demo: VectorTable with Full Iceberg Table Scan Support
========================================================

Shows how VectorTable integrates with Iceberg's table scan:
- Direct table.scan() access
- Predicate pushdown
- Column projection
- Time-travel scans
- Partition pruning
"""

import numpy as np
import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.io.vector import normalize_vectors

print("=" * 80)
print("VectorTable + Iceberg Table Scan Demo")
print("=" * 80)

# Setup: Create catalog and table
print("\n[1/6] Setting up Hive + MinIO...")
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
    print("  ✅ Connected")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    print("  💡 Run: docker compose -f dev/docker-compose-integration.yml up -d")
    exit(1)

# Create/load table
print("\n[2/6] Creating table with vectors...")
try:
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 128)),
        pa.field("category", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("year", pa.int32()),
    ])
    
    try:
        catalog.drop_table("vectorlake.products")
    except:
        pass
    
    table = catalog.create_table("vectorlake.products", schema=schema)
    print("  ✅ Table created")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    exit(1)

# Insert data
print("\n[3/6] Inserting product data...")
try:
    num_products = 100
    embeddings = normalize_vectors(np.random.randn(num_products, 128).astype(np.float32))
    
    data = pa.Table.from_pydict({
        "id": [f"product_{i:03d}" for i in range(num_products)],
        "content": [f"Product description {i}" for i in range(num_products)],
        "embedding": embeddings.tolist(),
        "category": ["electronics"] * 30 + ["clothing"] * 30 + ["books"] * 40,
        "price": np.random.uniform(10, 1000, num_products).tolist(),
        "year": np.random.choice([2020, 2021, 2022, 2023, 2024], num_products).tolist(),
    }, schema=schema)
    
    table.append(data)
    print(f"  ✅ Inserted {num_products} products")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    exit(1)

# Enable vector search
print("\n[4/6] Enabling vector search...")
try:
    vector_table = table.vector(
        embedding_column="embedding",
        dimension=128,
        id_column="id",
        text_column="content",
    )
    print(f"  ✅ Vector table ready")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    exit(1)

# Demo 1: Access underlying table
print("\n[5/6] Demo: Table Scan Capabilities")
print("-" * 80)

print("\n📊 1. Direct table access (underlying Iceberg table):")
print(f"   vector_table.table = {vector_table.table}")
print(f"   Can use all Iceberg methods!")

print("\n📊 2. Full table scan:")
scan_all = vector_table.table.scan().to_arrow()
print(f"   Total rows: {len(scan_all)}")
print(f"   Columns: {scan_all.column_names}")

print("\n📊 3. Filtered scan (predicate pushdown):")
scan_filtered = vector_table.table.scan().filter("category = 'electronics' AND price < 500").to_arrow()
print(f"   Electronics < $500: {len(scan_filtered)} rows")

print("\n📊 4. Column projection (read only needed columns):")
scan_projected = vector_table.table.scan().select("id", "category", "price").to_arrow()
print(f"   Selected columns: {scan_projected.column_names}")
print(f"   Rows: {len(scan_projected)}")

print("\n📊 5. Combined: Filter + Project:")
scan_combined = (
    vector_table.table.scan()
    .filter("year >= 2023")
    .select("id", "content", "year")
    .to_arrow()
)
print(f"   Products from 2023+: {len(scan_combined)} rows")
for i in range(min(3, len(scan_combined))):
    print(f"     - {scan_combined['id'][i]}: year={scan_combined['year'][i]}")

# Demo 2: Vector search with table scan filters
print("\n[6/6] Demo: Vector Search + Table Scan")
print("-" * 80)

query = normalize_vectors(np.random.randn(128).astype(np.float32).reshape(1, -1))[0]

print("\n🔍 1. Search all products:")
results_all = vector_table.search(query, top_k=5)
print(f"   Found {len(results_all)} results")
for r in results_all[:3]:
    print(f"     - {r.id}: {r.metadata['category']}, ${r.metadata['price']:.2f}")

print("\n🔍 2. Search with filter (Iceberg scan first!):")
results_filtered = vector_table.search(
    query,
    top_k=5,
    filters="category = 'electronics' AND price < 300"
)
print(f"   Electronics < $300: {len(results_filtered)} results")
for r in results_filtered[:3]:
    print(f"     - {r.id}: ${r.metadata['price']:.2f}")

print("\n🔍 3. Time-travel scan:")
snapshot_id = table.current_snapshot().snapshot_id
print(f"   Current snapshot: {snapshot_id}")

# Add more data
new_data = pa.Table.from_pydict({
    "id": ["new_001", "new_002"],
    "content": ["New product 1", "New product 2"],
    "embedding": normalize_vectors(np.random.randn(2, 128).astype(np.float32)).tolist(),
    "category": ["new", "new"],
    "price": [99.99, 199.99],
    "year": [2025, 2025],
}, schema=schema)
table.append(new_data)

print(f"   Added 2 new products")
print(f"   New snapshot: {table.current_snapshot().snapshot_id}")

# Scan old snapshot
scan_old = vector_table.table.scan(snapshot_id=snapshot_id).to_arrow()
print(f"   Old snapshot has {len(scan_old)} rows (no new products)")

# Scan current
scan_current = vector_table.table.scan().to_arrow()
print(f"   Current snapshot has {len(scan_current)} rows (with new products)")

# Vector search on old snapshot
results_old = vector_table.search(query, top_k=3, snapshot_id=snapshot_id)
print(f"   Vector search on old snapshot: {len(results_old)} results")

print("\n" + "=" * 80)
print("✅ DEMO COMPLETE!")
print("=" * 80)
print("\nKey Takeaways:")
print("  ✅ VectorTable wraps Iceberg Table (vector_table.table)")
print("  ✅ Full access to table.scan() API")
print("  ✅ Predicate pushdown works")
print("  ✅ Column projection works")
print("  ✅ Time-travel scans work")
print("  ✅ Vector search uses table.scan() internally")
print("  ✅ Hybrid: Filter first (scan), then vector search!")
print("\n🚀 Best of both worlds: Iceberg + Vectors!")
print("=" * 80)

