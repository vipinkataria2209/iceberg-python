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

"""Tests for vector I/O operations."""

import numpy as np
import pytest

from pyiceberg.io.vector import (
    VectorEmbedder,
    VectorIndex,
    normalize_vectors,
    select_index_strategy,
)

pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")


def test_vector_embedder():
    """Test VectorEmbedder basic functionality."""
    embedder = VectorEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    
    texts = ["Hello world", "PyIceberg is great"]
    embeddings = embedder.embed(texts)
    
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension
    assert embeddings.dtype == np.float32
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)


def test_vector_index_flat():
    """Test VectorIndex with flat index."""
    dimension = 128
    index = VectorIndex(dimension, index_type="Flat", metric="cosine")
    
    # Add vectors
    vectors = np.random.randn(100, dimension).astype(np.float32)
    vectors = normalize_vectors(vectors)
    index.add(vectors)
    
    assert index.ntotal == 100
    
    # Search
    query = np.random.randn(dimension).astype(np.float32)
    query = normalize_vectors(query.reshape(1, -1))[0]
    
    distances, indices = index.search(query, k=10)
    
    assert len(distances) == 10
    assert len(indices) == 10
    assert all(0 <= idx < 100 for idx in indices)


def test_vector_index_ivf():
    """Test VectorIndex with IVF index."""
    dimension = 128
    index = VectorIndex(dimension, index_type="IVF10,Flat", metric="cosine")
    
    # Add vectors (need enough for training)
    vectors = np.random.randn(1000, dimension).astype(np.float32)
    vectors = normalize_vectors(vectors)
    index.add(vectors)
    
    assert index.ntotal == 1000
    
    # Search
    query = normalize_vectors(np.random.randn(dimension).astype(np.float32).reshape(1, -1))[0]
    distances, indices = index.search(query, k=10)
    
    assert len(distances) == 10
    assert len(indices) == 10


def test_select_index_strategy():
    """Test automatic index strategy selection."""
    assert select_index_strategy(1000, 768) == "Flat"
    assert select_index_strategy(50000, 768) == "IVF223,Flat"
    assert select_index_strategy(500000, 768) == "IVF707,Flat"
    assert select_index_strategy(5000000, 768) == "IVF2236,Flat"


def test_normalize_vectors():
    """Test vector normalization."""
    vectors = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)
    normalized = normalize_vectors(vectors)
    
    expected = np.array([[0.6, 0.8], [5/13, 12/13]], dtype=np.float32)
    np.testing.assert_array_almost_equal(normalized, expected, decimal=5)
    
    # Check norms are 1
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)


def test_normalize_vectors_zero():
    """Test normalization handles zero vectors."""
    vectors = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    normalized = normalize_vectors(vectors)
    
    # Zero vector should remain zero
    assert np.all(normalized[0] == 0.0)
    
    # Non-zero vector should be normalized
    expected = np.array([0.6, 0.8], dtype=np.float32)
    np.testing.assert_array_almost_equal(normalized[1], expected, decimal=5)

