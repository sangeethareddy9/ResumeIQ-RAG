"""
src/scoring.py

Combines semantic similarity (from embeddings.py) with keyword-based
skill matching (from skills_data.py) into a single explainable match score.

Why combine both:
- Semantic similarity alone can miss the "did they literally mention AWS"
  signal that recruiters care about -- two resumes can be semantically
  similar to a JD for different reasons.
- Keyword matching alone misses synonyms and paraphrasing (e.g. "built
  REST APIs" vs "API development").
Combining them gives a score that's both meaning-aware and concretely
explainable -- which is exactly what this project is supposed to demonstrate.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.embeddings import embed_text, semantic_similarity
from src.skills_data import extract_skills_from_text

# Weighting: semantic similarity is the primary signal (captures meaning,
# paraphrasing, relevant experience described differently), keyword match
# is a secondary signal (concrete, explainable, catches exact requirements).
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


class MatchResult(BaseModel):
    """Full scoring breakdown for one resume against one job description."""

    filename: str
    semantic_score: float          # 0-100
    keyword_score: float           # 0-100
    final_score: float             # 0-100, weighted combination
    matched_skills: list[str]
    missing_skills: list[str]      # skills in the JD but not in the resume


def score_resume(jd_text: str, resume_text: str, filename: str) -> MatchResult:
    """
    Scores a single resume against a job description.

    jd_text and resume_text should both already be cleaned (see
    preprocess.clean_text) before being passed in here.
    """
    # --- Semantic similarity ---
    jd_embedding = embed_text(jd_text)
    resume_embedding = embed_text(resume_text)
    semantic = semantic_similarity(jd_embedding, resume_embedding)

    # --- Keyword overlap ---
    jd_skills = set(extract_skills_from_text(jd_text))
    resume_skills = set(extract_skills_from_text(resume_text))

    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)

    if jd_skills:
        keyword = (len(matched) / len(jd_skills)) * 100
    else:
        # JD mentioned no recognized skills from our vocabulary -- keyword
        # score is meaningless here, so we don't let it drag the final
        # score down; semantic similarity carries the full weight instead.
        keyword = semantic

    # --- Weighted final score ---
    final = (semantic * SEMANTIC_WEIGHT) + (keyword * KEYWORD_WEIGHT)

    return MatchResult(
        filename=filename,
        semantic_score=round(semantic, 2),
        keyword_score=round(keyword, 2),
        final_score=round(final, 2),
        matched_skills=matched,
        missing_skills=missing,
    )


def rank_resumes(jd_text: str, resumes: list[tuple[str, str]]) -> list[MatchResult]:
    """
    Scores and ranks multiple resumes against one JD.
    `resumes` is a list of (filename, cleaned_text) tuples.
    Returns results sorted by final_score, highest first.
    """
    results = [score_resume(jd_text, text, name) for name, text in resumes]
    results.sort(key=lambda r: r.final_score, reverse=True)
    return results