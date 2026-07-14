"""
src/skills_data.py

A curated vocabulary of common tech skills/tools used for keyword-based
explainability. This is intentionally a static list rather than something
dynamically generated -- it's simple, fast, and good enough for the
"explainability" layer the project needs (showing WHICH specific skills
matched, on top of the semantic similarity score).

Phase 5 polish idea: this could later be extended with spaCy's PhraseMatcher
for fuzzier matching (e.g. "AWS" matching "Amazon Web Services"), but a
straightforward substring list is the right starting point.
"""

COMMON_TECH_SKILLS = [
    # Languages
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
    "SQL", "R", "Scala", "Kotlin", "Swift", "PHP", "Ruby",
    # Web / Backend frameworks
    "FastAPI", "Django", "Flask", "React", "Vue", "Angular", "Node.js",
    "Express", "Spring Boot", "REST API", "GraphQL",
    # ML / Data
    "Machine Learning", "Deep Learning", "NLP", "PyTorch", "TensorFlow",
    "scikit-learn", "Pandas", "NumPy", "spaCy", "Hugging Face",
    "Sentence Transformers", "FAISS",
    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD",
    "GitHub Actions", "Jenkins", "Terraform",
    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    # General
    "Git", "Agile", "Microservices", "Linux",
]


import re


def extract_skills_from_text(text: str) -> list[str]:
    """
    Returns which skills from COMMON_TECH_SKILLS appear in the given text.

    Uses word-boundary regex matching rather than plain substring matching.
    Plain substring matching has a real bug: a skill like "R" or "Go" would
    match inside unrelated words ("FastAPI" contains... no, but "for"
    contains nothing, yet single letters like "R" match inside "REST",
    "Senior", "Director", etc). Word boundaries (\\b) ensure we only match
    whole words/phrases, not substrings buried inside other words.
    """
    found = []
    for skill in COMMON_TECH_SKILLS:
        # Escape special regex characters in the skill name (e.g. "C++"),
        # then wrap with word boundaries.
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(skill)
    return found