"""
src/vector_index.py

Wraps resume embeddings in a FAISS index for fast similarity search.

At small scale (a handful to a few dozen resumes), brute-force cosine
similarity (what scoring.py does) is already instant -- FAISS's real
advantage shows up at thousands+ of vectors, where a linear scan gets
slow. We still build it here because:
  1. It's part of the stated tech stack and a legitimate skill to
     demonstrate (interviewers will ask "why FAISS?" -- you can answer
     "for scalability beyond the demo dataset, and to practice the
     vector-search pattern I'd use in production").
  2. The interface (build an index once, query it many times) is exactly
     how a production version of this tool would behave once resume
     volume grows past what a Python list can comfortably scan.

We use IndexFlatIP (flat index, inner product) rather than an
approximate index (like IndexIVFFlat) because:
  - Flat = exact search, no accuracy tradeoff -- correct for this use case
    since we want precise rankings, not approximate ones.
  - sentence-transformers embeddings are NOT normalized by default, but
    inner product on L2-normalized vectors == cosine similarity. So we
    normalize vectors before adding them to the index.
"""

from __future__ import annotations

import faiss
import numpy as np


class ResumeVectorIndex:
    """
    Holds embeddings for a batch of resumes and supports fast similarity
    search against a query embedding (typically a job description).
    """

    def __init__(self, embedding_dim: int = 384):
        # IndexFlatIP = exact inner-product search. Combined with
        # L2-normalized vectors, inner product becomes cosine similarity.
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.filenames: list[str] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize so inner product == cosine similarity."""
        vectors = vectors.astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9  # avoid divide-by-zero for empty embeddings
        return vectors / norms

    def build(self, filenames: list[str], embeddings: np.ndarray) -> None:
        """
        Build the index from a batch of resume embeddings.
        embeddings shape: (n_resumes, embedding_dim) -- use embed_batch()
        from embeddings.py to produce this.
        """
        self.filenames = list(filenames)
        normalized = self._normalize(embeddings)
        self.index.add(normalized)

    def search(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[tuple[str, float]]:
        """
        Search the index with a query embedding (typically the JD embedding).
        Returns a list of (filename, similarity_score_0_to_100) tuples,
        sorted by similarity descending.

        top_k defaults to all indexed resumes (since for this project we
        want a full ranking, not just the top few matches).
        """
        if self.index.ntotal == 0:
            return []

        k = top_k or self.index.ntotal
        query = self._normalize(query_embedding.reshape(1, -1))

        # FAISS returns (distances, indices) -- for IndexFlatIP with
        # normalized vectors, "distance" here is actually the cosine
        # similarity (higher = more similar), despite the generic name.
        similarities, indices = self.index.search(query, k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if k > ntotal
                continue
            score = max(0.0, float(sim)) * 100
            results.append((self.filenames[idx], round(score, 2)))

        return results