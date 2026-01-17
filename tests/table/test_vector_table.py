# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests for VectorTable functionality."""

import numpy as np
import pyarrow as pa
import pytest

from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.table.vector import SearchResult, VectorTable

pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")


@pytest.fixture
def test_catalog(tmp_path) -> Catalog:
    """Create a test catalog."""
    return load_catalog(
        "test",
        **{
            "type": "sql",
            "uri": f"sqlite:///{tmp_path}/test.db",
            "warehouse": f"file://{tmp_path}",
        },
    )


@pytest.fixture
def embeddings_table(test_catalog: Catalog):
    """Create a test table with embeddings."""
    test_catalog.create_namespace("default")
    
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 384)),
        pa.field("category", pa.string()),
    ])
    
    table = test_catalog.create_table("default.embeddings", schema=schema)
    
    # Add some test data
    embeddings = np.random.randn(10, 384).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    data = pa.Table.from_pydict({
        "id": [f"doc_{i}" for i in range(10)],
        "text": [f"Document {i}" for i in range(10)],
        "embedding": embeddings.tolist(),
        "category": ["A"] * 5 + ["B"] * 5,
    })
    
    table.append(data)
    return table


def test_vector_table_init(embeddings_table):
    """Test VectorTable initialization."""
    vector_table = embeddings_table.vector(
        embedding_column="embedding",
        dimension=384,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    assert vector_table.dimension == 384
    assert vector_table.embedding_column == "embedding"
    assert vector_table.embedder is not None


def test_vector_table_search_with_vector(embeddings_table):
    """Test search with pre-computed vector."""
    vector_table = embeddings_table.vector(
        embedding_column="embedding",
        dimension=384
    )
    
    # Random query vector
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    results = vector_table.search(query, top_k=5)
    
    assert len(results) == 5
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(0 <= r.score <= 1 for r in results)
    # Results should be sorted by score (descending)
    assert results[0].score >= results[1].score


def test_vector_table_search_with_text(embeddings_table):
    """Test search with text query (auto-embedding)."""
    vector_table = embeddings_table.vector(
        embedding_column="embedding",
        dimension=384,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    results = vector_table.search("test query", top_k=5)
    
    assert len(results) == 5
    assert all(isinstance(r, SearchResult) for r in results)


def test_vector_table_search_with_filter(embeddings_table):
    """Test search with metadata filtering."""
    vector_table = embeddings_table.vector(
        embedding_column="embedding",
        dimension=384
    )
    
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    # Filter to only category A (5 docs)
    results = vector_table.search(query, top_k=10, filters="category = 'A'")
    
    assert len(results) == 5  # Only 5 docs in category A
    assert all(r.metadata["category"] == "A" for r in results)


def test_vector_table_add_documents(test_catalog: Catalog):
    """Test adding documents with auto-embedding."""
    test_catalog.create_namespace("test")
    
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 384)),
    ])
    
    table = test_catalog.create_table("test.docs", schema=schema)
    
    vector_table = table.vector(
        embedding_column="embedding",
        dimension=384,
        text_column="text",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Add documents
    vector_table.add_documents(
        documents=["First document", "Second document"],
        ids=["doc1", "doc2"]
    )
    
    # Verify data was added
    data = table.scan().to_arrow()
    assert len(data) == 2
    assert data["id"][0].as_py() == "doc1"
    assert data["text"][0].as_py() == "First document"


def test_vector_table_add_vectors(test_catalog: Catalog):
    """Test adding pre-computed vectors."""
    test_catalog.create_namespace("test2")
    
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 128)),
    ])
    
    table = test_catalog.create_table("test2.vecs", schema=schema)
    
    vector_table = table.vector(
        embedding_column="embedding",
        dimension=128
    )
    
    # Add vectors
    embeddings = np.random.randn(5, 128).astype(np.float32)
    ids = [f"vec_{i}" for i in range(5)]
    
    vector_table.add_vectors(ids, embeddings)
    
    # Verify
    data = table.scan().to_arrow()
    assert len(data) == 5


def test_search_result():
    """Test SearchResult dataclass."""
    result = SearchResult(
        id="doc1",
        score=0.95,
        text="Test document",
        metadata={"category": "A"},
        distance=0.05
    )
    
    assert result.id == "doc1"
    assert result.score == 0.95
    assert result.text == "Test document"
    assert result.metadata["category"] == "A"
    assert "doc1" in repr(result)

