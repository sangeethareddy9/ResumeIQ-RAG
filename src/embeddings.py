"""
src/embeddings.py

Generates sentence embeddings and computes semantic similarity between
a job description and resumes.

Model choice: all-MiniLM-L6-v2 is a small (~80MB), fast sentence-transformer
that's specifically trained for semantic similarity tasks. It's the standard
"good default" for this kind of project -- much faster than larger models,
while still producing high-quality embeddings for comparing JD vs resume text.

The model is loaded once at module level (same pattern as spaCy in
preprocess.py) since loading it repeatedly would be slow and wasteful --
each call would reload ~80MB from disk.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Loaded once, reused across every call. First load downloads the model
# (~80MB) and caches it locally -- subsequent runs are fast.
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> np.ndarray:
    """
    Convert a piece of text (job description or resume) into a single
    dense vector representing its meaning. Returns a numpy array of
    shape (384,) -- that's the embedding dimension for this model.
    """
    if not text or not text.strip():
        # Return a zero vector for empty text rather than crashing --
        # downstream code can check for this and flag it as unscoreable.
        return np.zeros(_model.get_sentence_embedding_dimension())

    embedding = _model.encode(text, convert_to_numpy=True)
    return embedding


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed multiple texts at once. Much faster than calling embed_text()
    in a loop -- the model batches the computation internally. Use this
    when embedding all resumes at once rather than one at a time.
    Returns shape (n_texts, 384).
    """
    if not texts:
        return np.zeros((0, _model.get_sentence_embedding_dimension()))

    # Replace empty strings so the model doesn't choke on them, but we
    # still get one embedding per input position.
    safe_texts = [t if t and t.strip() else " " for t in texts]
    embeddings = _model.encode(safe_texts, convert_to_numpy=True)
    return embeddings


def semantic_similarity(jd_embedding: np.ndarray, resume_embedding: np.ndarray) -> float:
    """
    Cosine similarity between a JD embedding and a resume embedding,
    scaled to a 0-100 score for display.

    Cosine similarity naturally ranges from -1 to 1, but in practice,
    semantically related text from the same domain (job descriptions and
    resumes) almost never produces negative values -- so we clip to 0
    as a safety floor rather than showing a confusing negative score.
    """
    sim = cosine_similarity(
        jd_embedding.reshape(1, -1),
        resume_embedding.reshape(1, -1),
    )[0][0]

    score = max(0.0, float(sim)) * 100
    return round(score, 2)