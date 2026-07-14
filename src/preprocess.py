"""
src/preprocess.py

Cleans raw extracted text before it goes into the embedding model.

Why this matters: resumes and job descriptions are full of formatting noise
-- extra whitespace, bullet characters, page artifacts -- that don't carry
meaning for semantic similarity. Cleaning it up gives the embedding model
cleaner input, which means better similarity scores later in Phase 2.

We use spaCy here for sentence-aware cleaning and (later) skill/keyword
extraction, rather than just doing raw string replacement.
"""

from __future__ import annotations

import re

import spacy

# Load once at module import time -- loading the model is slow (~1 sec),
# so we don't want to reload it every time clean_text() is called.
_nlp = spacy.load("en_core_web_sm")


def clean_text(raw_text: str) -> str:
    """
    Normalize whitespace, strip control characters, and collapse blank
    lines. Deliberately conservative: we don't remove punctuation or
    lowercase everything here, since sentence-transformers models are
    trained on natural, properly-cased text and do better with it.
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\x00", " ")

    # Collapse 3+ blank lines down to a single blank line
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Collapse runs of spaces/tabs (but keep newlines intact)
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace on each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def extract_skills(text: str, skill_vocabulary: list[str]) -> list[str]:
    """
    Very simple skill extraction: checks which terms from a known skill
    vocabulary appear in the text (case-insensitive). This is the
    explainability layer the project spec asks for -- it's deliberately
    simple now; Phase 2 will combine this with the semantic embedding
    score for the final weighted match score.

    `skill_vocabulary` is a flat list like ["Python", "React", "AWS", ...].
    A real version of this would pull the vocabulary from the JD itself --
    we'll build that in Phase 2.
    """
    text_lower = text.lower()
    found = []
    for skill in skill_vocabulary:
        if skill.lower() in text_lower:
            found.append(skill)
    return found


def get_doc(text: str):
    """
    Returns a spaCy Doc object for more advanced processing (sentence
    splitting, noun-phrase extraction, etc.) when needed in Phase 2.
    """
    return _nlp(text)