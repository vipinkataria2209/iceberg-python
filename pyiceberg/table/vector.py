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

"""Vector operations for Iceberg tables with semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from pyiceberg.io.vector import VectorEmbedder, VectorIndex, normalize_vectors, select_index_strategy

if TYPE_CHECKING:
    import pyarrow as pa

    from pyiceberg.table import Table


@dataclass
class SearchResult:
    """Result from vector similarity search.

    Attributes:
        id: Document ID
        score: Similarity score (0-1, higher is more similar)
        text: Document text (if available)
        metadata: Additional metadata fields
        distance: Raw distance from query
    """

    id: str
    score: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SearchResult(id={self.id!r}, score={self.score:.4f})"


class VectorTable:
    """Wrapper around Iceberg Table that adds vector search capabilities.

    This class enables semantic search on Iceberg tables containing vector embeddings.
    It supports:
    - Auto-embedding text to vectors
    - Efficient similarity search with FAISS
    - Hybrid filtering (metadata filters + vector search)
    - Time-travel queries on historical embeddings

    Example:
        >>> from pyiceberg.catalog import load_catalog
        >>> catalog = load_catalog("default")
        >>> table = catalog.load_table("docs.embeddings")
        >>>
        >>> # Enable vector search
        >>> vector_table = table.vector(
        ...     embedding_column="embedding",
        ...     dimension=768,
        ...     embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        ... )
        >>>
        >>> # Add documents with auto-embedding
        >>> vector_table.add_documents(
        ...     documents=["Apache Iceberg is a table format", "PyIceberg is a Python library"],
        ...     metadata=[{"category": "docs"}, {"category": "docs"}]
        ... )
        >>>
        >>> # Search
        >>> results = vector_table.search("table format for data lakes", top_k=10)
        >>> for result in results:
        ...     print(f"{result.id}: {result.score:.3f}")
    """

    def __init__(
        self,
        table: Table,
        embedding_column: str,
        dimension: int,
        id_column: str = "id",
        text_column: Optional[str] = None,
        embedding_model: Optional[str] = None,
        index_strategy: str = "auto",
        metric: str = "cosine",
    ):
        """Initialize VectorTable.

        Args:
            table: Underlying Iceberg table
            embedding_column: Name of column containing embeddings
            dimension: Embedding dimension
            id_column: Name of ID column (default: "id")
            text_column: Name of text column (optional)
            embedding_model: Model name for auto-embedding (optional)
            index_strategy: FAISS index type or "auto" (default: "auto")
            metric: Distance metric ("cosine" or "l2", default: "cosine")
        """
        self.table = table
        self.embedding_column = embedding_column
        self.dimension = dimension
        self.id_column = id_column
        self.text_column = text_column
        self.metric = metric

        # Auto-embedding setup
        self.embedder: Optional[VectorEmbedder] = None
        if embedding_model:
            self.embedder = VectorEmbedder(embedding_model)
            if self.embedder.dimension != dimension:
                raise ValueError(f"Model dimension {self.embedder.dimension} != specified dimension {dimension}")

        # Index setup
        self.index_strategy = index_strategy
        self.index: Optional[VectorIndex] = None
        self._index_stale = True
        self._cached_data: Optional[pa.Table] = None

    def _select_index_strategy(self) -> str:
        """Select optimal index strategy based on data size."""
        if self.index_strategy != "auto":
            return self.index_strategy

        # Estimate row count
        try:
            if self.table.current_snapshot():
                # Quick estimate from metadata
                row_count = sum(file.record_count for file in self.table.scan().plan_files())
            else:
                row_count = 0
        except Exception:
            row_count = 0

        return select_index_strategy(row_count, self.dimension)

    def _build_index(self, data: Optional[pa.Table] = None) -> None:
        """Build FAISS index from table data.

        Args:
            data: PyArrow table with embeddings (if None, scans from table)
        """
        import pyarrow as pa

        # Get data
        if data is None:
            data = self.table.scan().to_arrow()

        if len(data) == 0:
            # Empty table, create empty index
            strategy = self._select_index_strategy()
            self.index = VectorIndex(self.dimension, strategy, self.metric)
            self._index_stale = False
            self._cached_data = data
            return

        # Extract embeddings
        embeddings_col = data[self.embedding_column]

        # Convert to numpy array
        if pa.types.is_list(embeddings_col.type):
            # List of floats
            embeddings = np.array([row.as_py() for row in embeddings_col], dtype=np.float32)
        else:
            raise ValueError(f"Embedding column must be list type, got {embeddings_col.type}")

        # Normalize for cosine similarity
        if self.metric == "cosine":
            embeddings = normalize_vectors(embeddings)

        # Create and populate index
        strategy = self._select_index_strategy()
        self.index = VectorIndex(self.dimension, strategy, self.metric)
        self.index.add(embeddings)

        self._index_stale = False
        self._cached_data = data

    def add_documents(
        self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None, batch_size: int = 32
    ) -> None:
        """Add documents with automatic embedding.

        Args:
            documents: List of text documents to embed and add
            metadata: Optional list of metadata dicts (one per document)
            ids: Optional list of document IDs (auto-generated if None)
            batch_size: Batch size for embedding generation

        Example:
            >>> vector_table.add_documents(
            ...     documents=["First doc", "Second doc"],
            ...     metadata=[{"category": "A"}, {"category": "B"}],
            ...     ids=["doc1", "doc2"]
            ... )
        """
        import pyarrow as pa

        if not self.embedder:
            raise ValueError("No embedding model configured. Set embedding_model parameter in vector()")

        # Generate IDs if not provided
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        if len(ids) != len(documents):
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of documents ({len(documents)})")

        # Generate embeddings
        embeddings = self.embedder.embed(documents, batch_size=batch_size, normalize=(self.metric == "cosine"))

        # Build PyArrow table
        data: Dict[str, Any] = {self.id_column: ids, self.embedding_column: embeddings.tolist()}

        # Add text column if configured
        if self.text_column:
            data[self.text_column] = documents

        # Add metadata columns
        if metadata:
            if len(metadata) != len(documents):
                raise ValueError(f"Number of metadata dicts ({len(metadata)}) must match number of documents ({len(documents)})")

            # Extract all unique keys
            all_keys = set()
            for m in metadata:
                all_keys.update(m.keys())

            # Add each metadata field as a column
            for key in all_keys:
                data[key] = [m.get(key) for m in metadata]

        pa_table = pa.Table.from_pydict(data)

        # Append to Iceberg table
        self.table.append(pa_table)

        # Mark index as stale
        self._index_stale = True

    def add_vectors(
        self, ids: List[str], embeddings: Union[List[List[float]], np.ndarray], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add pre-computed vectors.

        Args:
            ids: List of document IDs
            embeddings: Pre-computed embeddings (list of lists or numpy array)
            metadata: Optional metadata dicts

        Example:
            >>> embeddings = np.random.randn(100, 768)
            >>> ids = [f"doc_{i}" for i in range(100)]
            >>> vector_table.add_vectors(ids, embeddings)
        """
        import pyarrow as pa

        # Convert to numpy if needed
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize if needed
        if self.metric == "cosine":
            embeddings = normalize_vectors(embeddings)

        # Build PyArrow table
        data: Dict[str, Any] = {self.id_column: ids, self.embedding_column: embeddings.tolist()}

        if metadata:
            all_keys = set()
            for m in metadata:
                all_keys.update(m.keys())
            for key in all_keys:
                data[key] = [m.get(key) for m in metadata]

        pa_table = pa.Table.from_pydict(data)
        self.table.append(pa_table)
        self._index_stale = True

    def search(
        self,
        query: Union[str, List[float], np.ndarray],
        top_k: int = 10,
        filters: Optional[str] = None,
        snapshot_id: Optional[int] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors.

        Args:
            query: Query text (auto-embedded) or vector
            top_k: Number of results to return
            filters: Iceberg filter expression (e.g., "category = 'electronics'")
            snapshot_id: Snapshot ID for time-travel search (optional)

        Returns:
            List of SearchResult objects ordered by similarity

        Example:
            >>> # Text query
            >>> results = vector_table.search("machine learning", top_k=10)
            >>>
            >>> # With filters
            >>> results = vector_table.search(
            ...     "machine learning",
            ...     top_k=10,
            ...     filters="category = 'ai' AND year >= 2020"
            ... )
            >>>
            >>> # Time-travel search
            >>> old_snapshot = table.history()[0].snapshot_id
            >>> results = vector_table.search("query", snapshot_id=old_snapshot)
        """
        # Convert query to vector
        if isinstance(query, str):
            if not self.embedder:
                raise ValueError("No embedding model configured for text queries. Set embedding_model parameter.")
            query_vector = self.embedder.embed([query], normalize=(self.metric == "cosine"))[0]
        else:
            query_vector = np.array(query, dtype=np.float32)
            if self.metric == "cosine":
                query_vector = normalize_vectors(query_vector.reshape(1, -1))[0]

        # Stage 1: Apply Iceberg filters (partition + file pruning)
        scan = self.table.scan(snapshot_id=snapshot_id)
        if filters:
            scan = scan.filter(filters)

        data = scan.to_arrow()

        if len(data) == 0:
            return []

        # Stage 2: Vector similarity search
        embeddings_col = data[self.embedding_column]
        embeddings = np.array([row.as_py() for row in embeddings_col], dtype=np.float32)

        # Normalize if needed
        if self.metric == "cosine":
            embeddings = normalize_vectors(embeddings)

        # Compute similarities
        if self.metric == "cosine":
            # Cosine similarity (inner product of normalized vectors)
            similarities = embeddings @ query_vector
            distances = 1 - similarities  # Convert to distance
        else:
            # L2 distance
            distances = np.linalg.norm(embeddings - query_vector, axis=1)
            similarities = 1 / (1 + distances)  # Convert to similarity score

        # Get top-k
        top_indices = np.argsort(-similarities)[: min(top_k, len(similarities))]

        # Build results
        results = []
        for idx in top_indices:
            idx_int = int(idx)
            result_dict = {
                "id": str(data[self.id_column][idx_int].as_py()),
                "score": float(similarities[idx_int]),
                "distance": float(distances[idx_int]),
            }

            if self.text_column and self.text_column in data.column_names:
                result_dict["text"] = str(data[self.text_column][idx_int].as_py())

            # Add metadata
            metadata = {}
            for col in data.column_names:
                if col not in [self.id_column, self.text_column, self.embedding_column]:
                    metadata[col] = data[col][idx_int].as_py()

            if metadata:
                result_dict["metadata"] = metadata

            results.append(SearchResult(**result_dict))

        return results

    def rebuild_index(self) -> None:
        """Force rebuild of FAISS index."""
        self._index_stale = True
        self._build_index()

    @property
    def num_vectors(self) -> int:
        """Get total number of vectors in the table."""
        if self.table.current_snapshot():
            return sum(file.file.record_count for file in self.table.scan().plan_files())
        return 0


# Extension method for Table class
def vector(
    self: Table,
    embedding_column: str,
    dimension: int,
    id_column: str = "id",
    text_column: Optional[str] = None,
    embedding_model: Optional[str] = None,
    index_strategy: str = "auto",
    metric: str = "cosine",
) -> VectorTable:
    """Enable vector search on this Iceberg table.

    Args:
        embedding_column: Name of column containing embeddings
        dimension: Embedding dimension
        id_column: Name of ID column (default: "id")
        text_column: Name of text column (optional)
        embedding_model: Model for auto-embedding (optional)
        index_strategy: FAISS index type or "auto" (default: "auto")
        metric: Distance metric ("cosine" or "l2", default: "cosine")

    Returns:
        VectorTable instance with search capabilities

    Example:
        >>> from pyiceberg.catalog import load_catalog
        >>> catalog = load_catalog("default")
        >>> table = catalog.load_table("docs.embeddings")
        >>>
        >>> # Enable vector search
        >>> vector_table = table.vector(
        ...     embedding_column="embedding",
        ...     dimension=768,
        ...     embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        ... )
        >>>
        >>> # Search
        >>> results = vector_table.search("PyIceberg features", top_k=5)
    """
    return VectorTable(
        table=self,
        embedding_column=embedding_column,
        dimension=dimension,
        id_column=id_column,
        text_column=text_column,
        embedding_model=embedding_model,
        index_strategy=index_strategy,
        metric=metric,
    )

