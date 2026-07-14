"""
src/test_embeddings.py

Full Phase 2 test: parse resumes, embed them, index with FAISS, query
the index with the JD, then enrich the FAISS ranking with keyword-based
explainability from scoring.py.

Run from project root:  python -m src.test_embeddings
"""

from pathlib import Path

from src.parser import parse_document
from src.preprocess import clean_text
from src.embeddings import embed_text, embed_batch
from src.vector_index import ResumeVectorIndex
from src.skills_data import extract_skills_from_text

SAMPLE_DIR = Path("data/sample_resumes")

JOB_DESCRIPTION = """
We are hiring a Senior Software Engineer to join our backend team.
Required skills: Python, FastAPI, REST API, cloud deployment (AWS),
Docker, and experience with CI/CD pipelines. Experience with React for
occasional frontend work is a plus. 3+ years of professional experience required.
"""

jd_clean = clean_text(JOB_DESCRIPTION)
jd_embedding = embed_text(jd_clean)
jd_skills = set(extract_skills_from_text(jd_clean))

# --- Parse + clean all resumes ---
files = list(SAMPLE_DIR.glob("*.pdf")) + list(SAMPLE_DIR.glob("*.docx"))

filenames = []
cleaned_texts = []
for file_path in files:
    file_bytes = file_path.read_bytes()
    parsed = parse_document(file_bytes, file_path.name)
    if parsed.is_usable:
        filenames.append(parsed.filename)
        cleaned_texts.append(clean_text(parsed.raw_text))
    else:
        print(f"Skipping {parsed.filename}: {parsed.status}")

# --- Embed all resumes in one batch call (faster than one at a time) ---
resume_embeddings = embed_batch(cleaned_texts)

# --- Build FAISS index and search it with the JD embedding ---
index = ResumeVectorIndex(embedding_dim=resume_embeddings.shape[1])
index.build(filenames, resume_embeddings)
ranked = index.search(jd_embedding)

# --- Enrich with keyword explainability ---
text_by_filename = dict(zip(filenames, cleaned_texts))

print("\n=== Ranked Results (FAISS-backed) ===\n")
for rank, (filename, semantic_score) in enumerate(ranked, start=1):
    resume_skills = set(extract_skills_from_text(text_by_filename[filename]))
    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    keyword_score = (len(matched) / len(jd_skills) * 100) if jd_skills else semantic_score
    final_score = round(semantic_score * 0.7 + keyword_score * 0.3, 2)

    print(f"{rank}. {filename}  --  Final score: {final_score}%")
    print(f"   Semantic (FAISS): {semantic_score}%  |  Keyword: {round(keyword_score, 2)}%")
    print(f"   Matched skills:  {', '.join(matched) or '(none)'}")
    print(f"   Missing skills:  {', '.join(missing) or '(none)'}")
    print()