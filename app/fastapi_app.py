"""
app/fastapi_app.py

REST API backend for the resume screener. Wraps the same scoring engine
used by the Streamlit app behind a proper HTTP endpoint.

POST /screen accepts a job description (as text) and multiple resume
files, returns a ranked JSON response with scores and skill breakdowns.

Run from project root with:
    uvicorn app.fastapi_app:app --reload --port 8000

Then test it at:
    http://127.0.0.1:8000/docs   (interactive Swagger UI, auto-generated)
"""

import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import parse_document
from src.preprocess import clean_text
from src.embeddings import embed_text, embed_batch
from src.vector_index import ResumeVectorIndex
from src.skills_data import extract_skills_from_text
from src.scoring import score_resume
from src.rag_assistant import (
    build_context_block,
    generate_advice,
    check_groundedness,
    OllamaConnectionError,
    AdviceGenerationError,
)

app = FastAPI(
    title="AI Resume Screener API",
    description="Semantic resume-to-job-description matching using sentence embeddings.",
    version="1.0.0",
)


class CandidateResult(BaseModel):
    filename: str
    match_score: float
    semantic_score: float
    keyword_score: float
    matched_skills: list[str]
    missing_skills: list[str]


class ScreenResponse(BaseModel):
    results: list[CandidateResult]
    skipped_files: list[str]


MAX_RESUMES = 20
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

# --- RAG Career Advisor: in-memory cache ---
# Keyed by filename -> cleaned resume text, populated whenever /screen runs.
# This lets /advise look up resume text by filename instead of requiring
# the resume file to be re-uploaded for every advice request.
#
# Known limitation (documented for portfolio/README purposes): this is
# process-local, in-memory state. It resets on server restart and isn't
# safe for multiple concurrent users sharing one server -- both are fine
# for a local single-user demo, but would need a real store (Redis, a
# database, or a session-scoped cache) in a production deployment.
_resume_text_cache: dict[str, str] = {}


class SkillGap(BaseModel):
    skill: str
    why_it_matters: str
    how_to_address: str


class InterviewQuestion(BaseModel):
    question: str
    related_skill: str
    purpose: str


class ResumeSuggestion(BaseModel):
    area: str
    suggestion: str
    rationale: str


class GapAnalysis(BaseModel):
    summary: str
    missing_skill_gaps: list[SkillGap]


class AdviseResponse(BaseModel):
    filename: str
    match_score: float
    gap_analysis: GapAnalysis
    interview_questions: list[InterviewQuestion]
    resume_suggestions: list[ResumeSuggestion]
    is_grounded: bool
    unverified_skills: list[str]


@app.get("/")
def root():
    """Simple health check -- confirms the API is running."""
    return {"status": "ok", "message": "AI Resume Screener API is running. See /docs for usage."}


@app.post("/screen", response_model=ScreenResponse)
async def screen_resumes(
    job_description: str = Form(..., description="The job description text"),
    resumes: list[UploadFile] = File(..., description="Resume files (PDF or DOCX)"),
):
    """
    Score and rank uploaded resumes against a job description.

    Validation:
    - Rejects empty job descriptions
    - Rejects unsupported file types (only .pdf/.docx allowed)
    - Caps at 20 resumes per request
    - Skips (rather than fails on) individual unreadable files -- a bad
      file in a batch shouldn't fail the whole request
    """
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty.")

    if not resumes:
        raise HTTPException(status_code=400, detail="At least one resume file is required.")

    if len(resumes) > MAX_RESUMES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(resumes)} uploaded, maximum is {MAX_RESUMES}.",
        )

    for f in resumes:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}' for file '{f.filename}'. Only .pdf and .docx are allowed.",
            )

    # --- Parse all resumes, skipping unreadable ones rather than failing ---
    filenames: list[str] = []
    cleaned_texts: list[str] = []
    skipped: list[str] = []

    for f in resumes:
        file_bytes = await f.read()
        parsed = parse_document(file_bytes, f.filename)
        if parsed.is_usable:
            filenames.append(parsed.filename)
            cleaned_texts.append(clean_text(parsed.raw_text))
        else:
            skipped.append(f"{parsed.filename} ({parsed.status.value})")

    if not filenames:
        raise HTTPException(
            status_code=422,
            detail="None of the uploaded files could be read as valid resumes.",
        )

    # Cache cleaned resume text by filename so /advise can retrieve it
    # later without requiring the resume file to be re-uploaded.
    for filename, cleaned in zip(filenames, cleaned_texts):
        _resume_text_cache[filename] = cleaned

    # --- Score using the same engine as the Streamlit app ---
    jd_clean = clean_text(job_description)
    jd_embedding = embed_text(jd_clean)
    jd_skills = set(extract_skills_from_text(jd_clean))

    resume_embeddings = embed_batch(cleaned_texts)

    index = ResumeVectorIndex(embedding_dim=resume_embeddings.shape[1])
    index.build(filenames, resume_embeddings)
    ranked = index.search(jd_embedding)

    text_by_filename = dict(zip(filenames, cleaned_texts))

    results = []
    for filename, semantic_score in ranked:
        resume_skills = set(extract_skills_from_text(text_by_filename[filename]))
        matched = sorted(jd_skills & resume_skills)
        missing = sorted(jd_skills - resume_skills)
        keyword_score = (len(matched) / len(jd_skills) * 100) if jd_skills else semantic_score
        final_score = round(semantic_score * 0.7 + keyword_score * 0.3, 2)

        results.append(CandidateResult(
            filename=filename,
            match_score=final_score,
            semantic_score=semantic_score,
            keyword_score=round(keyword_score, 2),
            matched_skills=matched,
            missing_skills=missing,
        ))

    return ScreenResponse(results=results, skipped_files=skipped)


@app.post("/advise", response_model=AdviseResponse)
async def advise(
    filename: str = Form(..., description="Filename of a resume previously screened via /screen"),
    job_description: str = Form(..., description="Job description text -- should match what was used for scoring"),
):
    """
    Generate AI career advice (gap analysis, interview questions, resume
    rewrite suggestions) for a previously-screened resume.

    Requires /screen to have been called first for this resume in the
    current server session -- resume text is looked up from the
    in-memory cache by filename rather than re-uploaded here.
    """
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty.")

    resume_text = _resume_text_cache.get(filename)
    if resume_text is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No cached resume text found for '{filename}'. "
                "Call /screen with this resume first, then request advice."
            ),
        )

    jd_clean = clean_text(job_description)
    match_result = score_resume(jd_clean, resume_text, filename)
    context = build_context_block(match_result, jd_clean, resume_text)

    try:
        advice_dict = generate_advice(context)
    except OllamaConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except AdviceGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    groundedness = check_groundedness(context, advice_dict)

    return AdviseResponse(
        filename=filename,
        match_score=match_result.final_score,
        gap_analysis=advice_dict["gap_analysis"],
        interview_questions=advice_dict["interview_questions"],
        resume_suggestions=advice_dict["resume_suggestions"],
        is_grounded=groundedness.is_grounded,
        unverified_skills=groundedness.unverified_skills,
    )