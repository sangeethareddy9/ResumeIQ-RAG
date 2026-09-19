"""
ResumeIQ - FastAPI Backend

REST API backend for the ResumeIQ AI Resume Screening System.

Endpoints:

POST /screen
    Accepts a job description and multiple PDF/DOCX resumes.
    Returns ATS scores, matched skills and missing skills.

POST /advise
    Generates grounded AI career advice for a resume that was
    previously screened.

The API uses the same parsing, scoring and RAG pipeline as the
Streamlit application.

Run from project root with:

    uvicorn app.fastapi_app:app --reload --port 8000

Swagger documentation:

    http://127.0.0.1:8000/docs
"""

import sys
from pathlib import Path
from typing import List

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from pydantic import BaseModel


# ---------------------------------------------------------
# Project Path
# ---------------------------------------------------------

sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parent
        .parent
    ),
)


# ---------------------------------------------------------
# ResumeIQ Imports
# ---------------------------------------------------------

from src.parser import parse_document
from src.preprocess import clean_text

from src.embeddings import (
    embed_text,
    embed_batch,
)

from src.vector_index import ResumeVectorIndex

from src.skills_data import (
    extract_skills_from_text,
)

from src.scoring import score_resume

from src.rag_assistant import (
    build_context_block,
    generate_advice,
    check_groundedness,
    AdviceGenerationError,
)


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="ResumeIQ API",
    description=(
        "AI resume screening, ATS scoring, semantic matching "
        "and grounded RAG career-advice API."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MAX_RESUMES = 20

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
}


# ---------------------------------------------------------
# Resume Screening Response Models
# ---------------------------------------------------------

class CandidateResult(BaseModel):
    filename: str
    match_score: float
    semantic_score: float
    keyword_score: float
    matched_skills: List[str]
    missing_skills: List[str]


class ScreenResponse(BaseModel):
    results: List[CandidateResult]
    skipped_files: List[str]


# ---------------------------------------------------------
# AI Advisor Response Models
# ---------------------------------------------------------

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
    missing_skill_gaps: List[SkillGap]


class AdviseResponse(BaseModel):
    filename: str
    match_score: float
    gap_analysis: GapAnalysis
    interview_questions: List[InterviewQuestion]
    resume_suggestions: List[ResumeSuggestion]
    is_grounded: bool
    unverified_skills: List[str]


# ---------------------------------------------------------
# In-Memory Resume Cache
# ---------------------------------------------------------

# /screen stores cleaned resume text here.
# /advise retrieves it using the filename.
#
# This is sufficient for the ResumeIQ demonstration.
# A production application should use persistent or
# session-scoped storage.

_resume_text_cache: dict[str, str] = {}


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/")
def root():
    """
    Confirm that the ResumeIQ API is running.
    """

    return {
        "status": "ok",
        "message": (
            "ResumeIQ API is running. "
            "Open /docs for Swagger documentation."
        ),
    }
# ---------------------------------------------------------
# Resume Screening Endpoint
# ---------------------------------------------------------

@app.post(
    "/screen",
    response_model=ScreenResponse,
)
async def screen_resumes(
    job_description: str = Form(
        ...,
        description="The job description text",
    ),
    resumes: List[UploadFile] = File(
        ...,
        description="Resume files (PDF or DOCX)",
        json_schema_extra={
            "items": {
                "type": "string",
                "format": "binary",
            }
        },
    ),
):

    """
    Score and rank uploaded resumes against a job description.

    Validation:
    - Reject empty job descriptions.
    - Allow only PDF and DOCX resumes.
    - Allow a maximum of 20 resumes.
    - Skip individual unreadable files instead of failing
      the entire batch.
    """

    # -----------------------------------------------------
    # Validate Job Description
    # -----------------------------------------------------

    if not job_description or not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="job_description cannot be empty.",
        )

    # -----------------------------------------------------
    # Validate Resume Uploads
    # -----------------------------------------------------

    if not resumes:
        raise HTTPException(
            status_code=400,
            detail="At least one resume file is required.",
        )

    if len(resumes) > MAX_RESUMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many files: {len(resumes)} uploaded. "
                f"Maximum allowed is {MAX_RESUMES}."
            ),
        )

    for resume_file in resumes:
        filename = resume_file.filename or ""

        suffix = Path(filename).suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{suffix}' "
                    f"for file '{filename}'. "
                    "Only .pdf and .docx are allowed."
                ),
            )

    # -----------------------------------------------------
    # Parse Resume Files
    # -----------------------------------------------------

    filenames: List[str] = []
    cleaned_texts: List[str] = []
    skipped_files: List[str] = []

    for resume_file in resumes:
        filename = (
            resume_file.filename
            or "unknown_resume"
        )

        try:
            file_bytes = await resume_file.read()

            parsed = parse_document(
                file_bytes,
                filename,
            )

        except Exception:
            skipped_files.append(
                f"{filename} (parsing failed)"
            )
            continue

        if parsed.is_usable:
            cleaned_resume = clean_text(
                parsed.raw_text
            )

            filenames.append(
                parsed.filename
            )

            cleaned_texts.append(
                cleaned_resume
            )

        else:
            skipped_files.append(
                (
                    f"{parsed.filename} "
                    f"({parsed.status.value})"
                )
            )

    # -----------------------------------------------------
    # Ensure At Least One Resume Was Parsed
    # -----------------------------------------------------

    if not filenames:
        raise HTTPException(
            status_code=422,
            detail=(
                "None of the uploaded files could be "
                "read as valid resumes."
            ),
        )

    # -----------------------------------------------------
    # Cache Resume Text for /advise
    # -----------------------------------------------------

    for filename, cleaned_resume in zip(
        filenames,
        cleaned_texts,
    ):
        _resume_text_cache[
            filename
        ] = cleaned_resume

    # -----------------------------------------------------
    # Preprocess Job Description
    # -----------------------------------------------------

    jd_clean = clean_text(
        job_description
    )

    # -----------------------------------------------------
    # Generate Embeddings
    # -----------------------------------------------------

    try:
        jd_embedding = embed_text(
            jd_clean
        )

        resume_embeddings = embed_batch(
            cleaned_texts
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate resume embeddings."
            ),
        ) from exc

    # -----------------------------------------------------
    # Extract JD Skills
    # -----------------------------------------------------

    jd_skills = set(
        extract_skills_from_text(
            jd_clean
        )
    )

    # -----------------------------------------------------
    # Build Vector Index and Rank Resumes
    # -----------------------------------------------------

    try:
        index = ResumeVectorIndex(
            embedding_dim=(
                resume_embeddings.shape[1]
            )
        )

        index.build(
            filenames,
            resume_embeddings,
        )

        ranked = index.search(
            jd_embedding
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to rank resumes.",
        ) from exc

    # -----------------------------------------------------
    # Map Filename to Resume Text
    # -----------------------------------------------------

    text_by_filename = dict(
        zip(
            filenames,
            cleaned_texts,
        )
    )

    # -----------------------------------------------------
    # Calculate ATS Results
    # -----------------------------------------------------

    results: List[CandidateResult] = []

    for filename, semantic_score in ranked:

        resume_text = text_by_filename[
            filename
        ]

        resume_skills = set(
            extract_skills_from_text(
                resume_text
            )
        )

        matched_skills = sorted(
            jd_skills & resume_skills
        )

        missing_skills = sorted(
            jd_skills - resume_skills
        )

        if jd_skills:
            keyword_score = (
                len(matched_skills)
                / len(jd_skills)
                * 100
            )
        else:
            keyword_score = semantic_score

        final_score = round(
            (
                semantic_score * 0.7
                + keyword_score * 0.3
            ),
            2,
        )

        results.append(
            CandidateResult(
                filename=filename,
                match_score=final_score,
                semantic_score=semantic_score,
                keyword_score=round(
                    keyword_score,
                    2,
                ),
                matched_skills=matched_skills,
                missing_skills=missing_skills,
            )
        )

    # -----------------------------------------------------
    # Return Results
    # -----------------------------------------------------

    return ScreenResponse(
        results=results,
        skipped_files=skipped_files,
    )
# ---------------------------------------------------------
# AI Career Advisor Endpoint
# ---------------------------------------------------------

@app.post(
    "/advise",
    response_model=AdviseResponse,
)
async def advise(
    filename: str = Form(
        ...,
        description=(
            "Filename of a resume previously "
            "screened via /screen"
        ),
    ),
    job_description: str = Form(
        ...,
        description=(
            "Job description text used for scoring"
        ),
    ),
):
    """
    Generate grounded AI career advice for a resume
    previously screened through /screen.

    Workflow:
    - Retrieve cached resume text.
    - Calculate ATS score.
    - Build RAG context.
    - Generate structured advice using Groq.
    - Validate structured output.
    - Perform groundedness verification.
    """

    # -----------------------------------------------------
    # Validate Inputs
    # -----------------------------------------------------

    if not filename or not filename.strip():
        raise HTTPException(
            status_code=400,
            detail="filename cannot be empty.",
        )

    if (
        not job_description
        or not job_description.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "job_description cannot be empty."
            ),
        )

    # -----------------------------------------------------
    # Retrieve Previously Screened Resume
    # -----------------------------------------------------

    resume_text = _resume_text_cache.get(
        filename
    )

    if resume_text is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No cached resume text was found for "
                f"'{filename}'. Run /screen with this "
                "resume before requesting AI advice."
            ),
        )

    # -----------------------------------------------------
    # Preprocess Job Description
    # -----------------------------------------------------

    jd_clean = clean_text(
        job_description
    )

    # -----------------------------------------------------
    # ATS Scoring
    # -----------------------------------------------------

    try:
        match_result = score_resume(
            jd_clean,
            resume_text,
            filename,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to calculate the ATS score."
            ),
        ) from exc

    # -----------------------------------------------------
    # Build RAG Context
    # -----------------------------------------------------

    try:
        context = build_context_block(
            match_result,
            jd_clean,
            resume_text,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to build the RAG context."
            ),
        ) from exc

    # -----------------------------------------------------
    # Generate Structured AI Advice
    # -----------------------------------------------------

    try:
        advice = generate_advice(
            context
        )

    except AdviceGenerationError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The AI Career Advisor is temporarily "
                "unavailable. Please try again."
            ),
        ) from exc

    # -----------------------------------------------------
    # Groundedness Verification
    # -----------------------------------------------------

    try:
        groundedness = check_groundedness(
            context,
            advice,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to verify AI groundedness."
            ),
        ) from exc

        # -----------------------------------------------------
    # Return Structured JSON Response
    # -----------------------------------------------------

    return AdviseResponse(
        filename=filename,
        match_score=match_result.final_score,
        gap_analysis=(
            advice.gap_analysis.model_dump()
        ),
        interview_questions=[
            item.model_dump()
            for item in advice.interview_questions
        ],
        resume_suggestions=[
            item.model_dump()
            for item in advice.resume_suggestions
        ],
        is_grounded=(
            groundedness.is_grounded
        ),
        unverified_skills=(
            groundedness.unverified_skills
        ),
    )