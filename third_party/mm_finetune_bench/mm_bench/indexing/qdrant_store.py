"""Qdrant-backed vector store for encoded keyframes.

One collection per (model, variant) pair -- e.g. ``aic_pe_core_bigg_v1`` for
the pretrained baseline and ``aic_pe_core_bigg_v2`` after fine-tuning --
since embedding dimensions and, more importantly, the embedding *space*
itself differ per model and per fine-tune version; mixing them in one
collection would make distances meaningless.

Point IDs are a deterministic 63-bit hash of the image's relative path, so
re-running an interrupted encode upserts over the same points rather than
duplicating them. That is what makes ``--resume`` correct without needing a
separate progress file: the set of IDs already present in the collection
*is* the progress record.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def point_id_for_path(image_path: str) -> int:
    """Stable, collision-resistant 63-bit point ID derived from the image
    path (Qdrant accepts unsigned ints; staying under 2**63 avoids any
    signed/unsigned ambiguity in clients along the way).
    """
    digest = hashlib.blake2b(image_path.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


class QdrantVectorStore:
    def __init__(
        self,
        collection: str,
        vector_size: int,
        url: str = "http://localhost:6333",
        distance: str = "cosine",
        timeout: int = 120,
    ):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.vector_size = vector_size
        self.distance = distance
        self.client = QdrantClient(url=url, timeout=timeout)

    # -- collection lifecycle ------------------------------------------------
    def ensure_collection(self, recreate: bool = False) -> None:
        from qdrant_client import models

        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            info = self.client.get_collection(self.collection)
            existing_size = info.config.params.vectors.size
            if existing_size != self.vector_size:
                raise ValueError(
                    f"Collection '{self.collection}' already exists with vector size "
                    f"{existing_size}, but this model emits {self.vector_size}-d vectors. "
                    f"Pass --recreate to drop and rebuild it."
                )
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE if self.distance == "cosine" else models.Distance.DOT,
            ),
            # Keyframe corpora are write-heavy during the initial encode and
            # read-only afterwards; deferring HNSW construction until the
            # bulk load is done is substantially faster than building it
            # incrementally per upsert batch.
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )

    def finalize_index(self, indexing_threshold: int = 20000) -> None:
        """Re-enable HNSW construction after a bulk load."""
        from qdrant_client import models

        self.client.update_collection(
            collection_name=self.collection,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=indexing_threshold),
        )

    # -- writes ------------------------------------------------------------
    def upsert_batch(
        self,
        image_paths: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[Dict[str, Any]],
        wait: bool = False,
    ) -> None:
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=point_id_for_path(path),
                vector=vec.tolist(),
                payload=payload,
            )
            for path, vec, payload in zip(image_paths, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=wait)

    # -- reads ------------------------------------------------------------
    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def existing_ids(self, candidate_ids: Sequence[int], chunk: int = 1000) -> set:
        """Return the subset of ``candidate_ids`` already stored.

        Used by ``--resume`` to skip work. Queried in chunks because a
        retrieve() call with hundreds of thousands of IDs would blow past
        the HTTP request size limit.
        """
        found = set()
        for i in range(0, len(candidate_ids), chunk):
            batch = list(candidate_ids[i : i + chunk])
            records = self.client.retrieve(
                collection_name=self.collection, ids=batch, with_payload=False, with_vectors=False
            )
            found.update(r.id for r in records)
        return found

    def search(self, query_vector: np.ndarray, limit: int = 10, with_payload: bool = True):
        return self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=with_payload,
        ).points
