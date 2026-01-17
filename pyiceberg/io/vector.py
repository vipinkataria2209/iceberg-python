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

"""Vector I/O operations for embeddings and similarity search."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

# Optional dependencies - check availability without importing
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None  # type: ignore

# Lazy import for sentence_transformers (avoid eager loading)
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    import importlib.util
    if importlib.util.find_spec("sentence_transformers") is not None:
        SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    pass


class VectorEmbedder:
    """Generate embeddings from text using sentence transformers.

    Example:
        >>> embedder = VectorEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        >>> vectors = embedder.embed(["Hello world", "PyIceberg rocks"])
        >>> vectors.shape
        (2, 384)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedder with a model.

        Args:
            model_name: HuggingFace model name or path
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for auto-embedding. "
                "Install with: pip install 'pyiceberg[vector]'"
            )
        # Lazy import
        from sentence_transformers import SentenceTransformer
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """Embed texts to vectors.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings

        Returns:
            Array of embeddings with shape (len(texts), dimension)
        """
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=normalize)
        return embeddings


class VectorIndex:
    """FAISS-based vector index for similarity search.

    Example:
        >>> index = VectorIndex(dimension=384, index_type="Flat")
        >>> vectors = np.random.randn(1000, 384).astype(np.float32)
        >>> index.add(vectors)
        >>> query = np.random.randn(384).astype(np.float32)
        >>> distances, indices = index.search(query, k=10)
    """

    def __init__(self, dimension: int, index_type: str = "Flat", metric: str = "cosine"):
        """Initialize FAISS index.

        Args:
            dimension: Embedding dimension
            index_type: FAISS index type (e.g., "Flat", "IVF100,Flat", "IVF1000,PQ32")
            metric: Distance metric ("cosine" or "l2")
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu or faiss-gpu is required for vector search. " "Install with: pip install 'pyiceberg[vector]'")

        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.index = self._create_index()
        self._num_vectors = 0

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        # Choose metric
        if self.metric == "cosine":
            metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "l2":
            metric = faiss.METRIC_L2
        else:
            raise ValueError(f"Unknown metric: {self.metric}. Use 'cosine' or 'l2'")

        # Create index
        if self.index_type == "Flat":
            if metric == faiss.METRIC_INNER_PRODUCT:
                return faiss.IndexFlatIP(self.dimension)
            else:
                return faiss.IndexFlatL2(self.dimension)

        elif self.index_type.startswith("IVF"):
            # Parse IVF config (e.g., "IVF100,Flat")
            parts = self.index_type.split(",")
            nlist = int(parts[0].replace("IVF", ""))

            if len(parts) == 1 or parts[1] == "Flat":
                quantizer = faiss.IndexFlatIP(self.dimension) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dimension)
                return faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric)
            else:
                # Use factory for complex indexes (e.g., IVF1000,PQ32)
                return faiss.index_factory(self.dimension, self.index_type, metric)
        else:
            # General factory
            return faiss.index_factory(self.dimension, self.index_type, metric)

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.

        Args:
            vectors: Numpy array of shape (n_vectors, dimension)
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Train index if needed (IVF requires training)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(vectors)

        self.index.add(vectors)
        self._num_vectors += len(vectors)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector of shape (dimension,)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices) arrays of shape (k,)
        """
        if query.dtype != np.float32:
            query = query.astype(np.float32)

        # Ensure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances, indices = self.index.search(query, min(k, self._num_vectors))
        return distances[0], indices[0]

    @property
    def ntotal(self) -> int:
        """Total number of vectors in the index."""
        return self.index.ntotal


def select_index_strategy(num_vectors: int, dimension: int) -> str:
    """Auto-select optimal FAISS index based on dataset size.

    Args:
        num_vectors: Number of vectors
        dimension: Embedding dimension

    Returns:
        FAISS index string (e.g., "Flat", "IVF1000,Flat")

    Example:
        >>> select_index_strategy(1000, 768)
        'Flat'
        >>> select_index_strategy(100000, 768)
        'IVF316,Flat'
        >>> select_index_strategy(10000000, 768)
        'IVF4096,Flat'
    """
    if num_vectors < 10_000:
        return "Flat"
    elif num_vectors < 100_000:
        nlist = min(100, int(np.sqrt(num_vectors)))
        return f"IVF{nlist},Flat"
    elif num_vectors < 1_000_000:
        nlist = min(1000, int(np.sqrt(num_vectors)))
        return f"IVF{nlist},Flat"
    elif num_vectors < 10_000_000:
        nlist = min(4096, int(np.sqrt(num_vectors)))
        return f"IVF{nlist},Flat"
    else:
        # For very large datasets, consider Product Quantization
        nlist = 10000
        m = dimension // 8
        return f"IVF{nlist},PQ{m}"


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors for cosine similarity.

    Args:
        vectors: Array of shape (n_vectors, dimension)

    Returns:
        Normalized vectors of same shape
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms

