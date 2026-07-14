"""
Quick manual smoke test for Phase 1 -- src/rag_assistant.py.
Run with: python smoke_test_rag.py
"""

from src.parser import parse_document
from src.preprocess import clean_text
from src.scoring import score_resume
from src.rag_assistant import build_context_block

# A short sample JD -- edit this to whatever you like
jd_raw = """
We are looking for a Python developer with experience in AWS,
Docker, and FastAPI. Familiarity with PostgreSQL and CI/CD pipelines
is a plus.
"""

with open("data/sample_resumes/john_doe.docx", "rb") as f:
    file_bytes = f.read()

parsed = parse_document(file_bytes, "john_doe.docx")
if not parsed.is_usable:
    raise SystemExit(f"Could not parse resume: {parsed.status} -- {parsed.error_message}")

jd_clean = clean_text(jd_raw)
resume_clean = clean_text(parsed.raw_text)

result = score_resume(jd_clean, resume_clean, parsed.filename)
context = build_context_block(result, jd_clean, resume_clean)

print(context.as_prompt_text())