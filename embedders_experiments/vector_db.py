from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

logger = logging.getLogger(__name__)


class VectorDBCollection(ABC):
    """Abstract interface for vector database collections."""

    @abstractmethod
    def add(
            self,
            documents: Sequence[str],
            ids: Sequence[str],
            metadatas: Sequence[Mapping[str, str]],
    ) -> None:
        """Add documents to the collection."""
        pass

    @abstractmethod
    def query(
            self,
            query_texts: List[str] | None = None,
            query_embeddings: List[List[float]] | None = None,
            n_results: int = 10,
    ) -> Dict[str, Any]:
        """Query the collection.

        Returns a dict with keys: 'ids', 'distances', 'documents', 'metadatas'
        Each value is a list of lists (one list per query).
        """
        pass


class ChromaDBCollection(VectorDBCollection):
    """Wrapper for ChromaDB collection."""

    def __init__(self, collection):
        self._collection = collection

    def add(
            self,
            documents: Sequence[str],
            ids: Sequence[str],
            metadatas: Sequence[Mapping[str, str]],
    ) -> None:
        self._collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def query(
            self,
            query_texts: List[str] | None = None,
            query_embeddings: List[List[float]] | None = None,
            n_results: int = 10,
    ) -> Dict[str, Any]:
        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
        )


class QdrantCollection(VectorDBCollection):
    """Wrapper for Qdrant collection."""

    def __init__(self, client, collection_name: str, embedder):
        from qdrant_client.models import Distance, VectorParams

        self._client = client
        self._collection_name = collection_name
        self._embedder = embedder
        self._next_id = 0

    def add(
            self,
            documents: Sequence[str],
            ids: Sequence[str],
            metadatas: Sequence[Mapping[str, str]],
    ) -> None:
        from qdrant_client.models import PointStruct

        # Generate embeddings for documents
        if hasattr(self._embedder, '__call__'):
            embeddings = self._embedder(documents)
        else:
            raise ValueError("Embedder must be callable")

        # Create points
        points = []
        for doc, doc_id, meta, embedding in zip(documents, ids, metadatas, embeddings):
            # Convert embedding to list if it's a numpy array
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()

            point = PointStruct(
                id=self._next_id,
                vector=embedding,
                payload={
                    "document": doc,
                    "id": doc_id,
                    **meta,
                }
            )
            points.append(point)
            self._next_id += 1

        # Upload points
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    def query(
            self,
            query_texts: List[str] | None = None,
            query_embeddings: List[List[float]] | None = None,
            n_results: int = 10,
    ) -> Dict[str, Any]:
        # Generate query embeddings if needed
        if query_embeddings is None and query_texts is not None:
            if hasattr(self._embedder, '__call__'):
                query_embeddings = self._embedder(query_texts)
            else:
                raise ValueError("Embedder must be callable for text queries")

        if query_embeddings is None:
            raise ValueError("Either query_texts or query_embeddings must be provided")

        # Query Qdrant
        all_ids = []
        all_distances = []
        all_documents = []
        all_metadatas = []

        for query_embedding in query_embeddings:
            # Convert embedding to list if it's a numpy array
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()

            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                limit=n_results,
                with_payload=True,
            ).points

            ids = []
            distances = []
            documents = []
            metadatas = []

            for result in results:
                # Access payload from the ScoredPoint object
                payload = result.payload if hasattr(result, 'payload') else {}
                ids.append(payload.get("id", ""))
                distances.append(result.score)
                documents.append(payload.get("document", ""))
                # Extract metadata (excluding document and id)
                meta = {k: v for k, v in payload.items() if k not in {"document", "id"}}
                metadatas.append(meta)

            all_ids.append(ids)
            all_distances.append(distances)
            all_documents.append(documents)
            all_metadatas.append(metadatas)

        return {
            "ids": all_ids,
            "distances": all_distances,
            "documents": all_documents,
            "metadatas": all_metadatas,
        }


def create_vector_db(
        db_type: str,
        database: str,
        collection_name: str,
        embedder,
        distance_metric: str = "cosine",
) -> VectorDBCollection:
    """Create a vector database collection.

    Args:
        db_type: Type of database ("chroma" or "qdrant")
        database: Database path or connection string
        collection_name: Name of the collection
        embedder: Embedding function
        distance_metric: Distance metric ("cosine", "l2", "ip")

    Returns:
        VectorDBCollection instance
    """
    if db_type == "chroma":
        import chromadb

        # Validate distance metric
        valid_metrics = {"cosine", "l2", "ip"}
        if distance_metric not in valid_metrics:
            raise ValueError(
                f"Invalid distance_metric: {distance_metric}. "
                f"Must be one of: {valid_metrics}"
            )

        persist_dir = None if database == ":memory:" or database == "" else database
        if persist_dir:
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            # Recreate the collection each run to avoid ID collisions
            try:
                chroma_client.delete_collection(collection_name)
            except Exception:
                pass
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedder,
                metadata={"hnsw:space": distance_metric},
            )
        else:
            chroma_client = chromadb.Client()
            collection = chroma_client.create_collection(
                name="inmemory",
                embedding_function=embedder,
                metadata={"hnsw:space": distance_metric},
            )

        logger.info(f"Using ChromaDB with distance metric: {distance_metric}")
        return ChromaDBCollection(collection)

    elif db_type == "qdrant":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        # Map distance metric to Qdrant distance
        distance_map = {
            "cosine": Distance.COSINE,
            "l2": Distance.EUCLID,
            "ip": Distance.DOT,
        }

        if distance_metric not in distance_map:
            raise ValueError(
                f"Invalid distance_metric: {distance_metric}. "
                f"Must be one of: {list(distance_map.keys())}"
            )

        qdrant_distance = distance_map[distance_metric]

        # Create Qdrant client
        if database == ":memory:" or database == "":
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(path=database)

        # Get embedding dimension
        test_embedding = embedder(["test"])[0]
        vector_size = len(test_embedding)

        # Recreate collection
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=qdrant_distance),
        )

        logger.info(f"Using Qdrant with distance metric: {distance_metric}")
        return QdrantCollection(client, collection_name, embedder)

    else:
        raise ValueError(f"Unknown database type: {db_type}. Must be 'chroma' or 'qdrant'")