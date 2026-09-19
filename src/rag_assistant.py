"""
ResumeIQ - AI Resume Screening System

This module is responsible for:

1. Resume text chunking for RAG.
2. Building Retrieval Augmented Generation context using FAISS.
3. Sending grounded context to Groq.
4. Generating AI career advice.
5. Performing groundedness verification.
6. Supporting chat-based follow-up questions.

The AI should only reason over evidence produced by the ATS pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer

from src.scoring import MatchResult


# ---------------------------------------------------------
# Environment / Logging
# ---------------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Add GROQ_API_KEY to your .env file or deployment secrets."
    )


# ---------------------------------------------------------
# Groq
# ---------------------------------------------------------

client = Groq(api_key=GROQ_API_KEY)

LLM_MODEL = "openai/gpt-oss-20b"


# ---------------------------------------------------------
# RAG Configuration
# ---------------------------------------------------------

MAX_RAW_TEXT_CHARS = 6000
MAX_CHUNKS_FOR_RAG = 5
CHUNK_SIZE = 400


# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------

try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully.")

except Exception as exc:
    logger.error("Failed to load embedding model: %s", exc)
    embed_model = None


# ---------------------------------------------------------
# Structured Output Models
# ---------------------------------------------------------

class GapItem(BaseModel):
    skill: str = Field(
        ...,
        description="Missing skill name",
    )

    why_it_matters: str = Field(
        ...,
        description="Why the skill matters for this job",
    )

    how_to_address: str = Field(
        ...,
        description="Concrete action to address the skill gap",
    )


class GapAnalysis(BaseModel):
    summary: str

    missing_skill_gaps: List[GapItem] = Field(
        default_factory=list
    )


class InterviewQ(BaseModel):
    question: str = Field(
        ...,
        description="Interview question",
    )

    related_skill: str = Field(
        ...,
        description="Skill tested by the question",
    )

    purpose: str = Field(
        ...,
        description="Purpose of the question",
    )


class ResumeSuggestion(BaseModel):
    area: str = Field(
        ...,
        description="Resume area to improve",
    )

    suggestion: str = Field(
        ...,
        description="Concrete resume suggestion",
    )

    rationale: str = Field(
        ...,
        description="Reason for the suggestion",
    )


class AdviceOutput(BaseModel):
    gap_analysis: GapAnalysis

    interview_questions: List[InterviewQ] = Field(
        ...,
        min_length=3,
        max_length=5,
    )

    resume_suggestions: List[ResumeSuggestion] = Field(
        ...,
        min_length=3,
        max_length=3,
    )


# ---------------------------------------------------------
# RAG Context
# ---------------------------------------------------------

class ContextBlock(BaseModel):
    """
    Evidence supplied to the AI Advisor.

    All candidate-specific information comes from the
    ATS pipeline, resume and job description.
    """

    filename: str

    final_score: float
    semantic_score: float
    keyword_score: float

    matched_skills: list[str]
    missing_skills: list[str]

    jd_text: str
    resume_text: str

    resume_chunks: list[str] = Field(default_factory=list)

    def as_prompt_text(self) -> str:
        matched = ", ".join(self.matched_skills) or "(none)"
        missing = ", ".join(self.missing_skills) or "(none)"

        if self.resume_chunks:
            rag_context = "\n".join(
                f"- {chunk}"
                for chunk in self.resume_chunks
            )
        else:
            rag_context = "(no relevant chunks)"

        return f"""
CANDIDATE
---------
{self.filename}

OVERALL MATCH SCORE
-------------------
{self.final_score:.2f}/100

SEMANTIC SCORE
--------------
{self.semantic_score:.2f}%

KEYWORD SCORE
-------------
{self.keyword_score:.2f}%

MATCHED SKILLS
--------------
{matched}

MISSING SKILLS
--------------
{missing}

JOB DESCRIPTION
---------------
{self.jd_text}

RELEVANT RESUME CONTEXT - RAG RETRIEVAL
---------------------------------------
{rag_context}

FULL RESUME
-----------
{self.resume_text}
"""
# ---------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
) -> List[str]:
    """
    Split parsed resume text into chunks while trying
    to preserve sentence boundaries.
    """

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            f"{current_chunk} {sentence}".strip()
            if current_chunk
            else sentence
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)

        # Handle a single sentence larger than chunk_size.
        if len(sentence) > chunk_size:
            for start in range(
                0,
                len(sentence),
                chunk_size,
            ):
                piece = sentence[
                    start:start + chunk_size
                ].strip()

                if piece:
                    chunks.append(piece)

            current_chunk = ""

        else:
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---------------------------------------------------------
# FAISS Retrieval
# ---------------------------------------------------------

def retrieve_relevant_chunks(
    query: str,
    chunks: List[str],
    k: int = MAX_CHUNKS_FOR_RAG,
) -> List[str]:
    """
    Retrieve resume chunks most relevant to the job
    description.

    SentenceTransformer generates embeddings.
    FAISS performs similarity retrieval using normalized
    vectors and inner product.
    """

    if not chunks:
        return []

    if not query or not query.strip():
        return chunks[:k]

    if embed_model is None:
        logger.warning(
            "Embedding model unavailable. "
            "Returning the first %s chunks.",
            k,
        )
        return chunks[:k]

    try:
        # Create embeddings for resume chunks.
        chunk_embeddings = embed_model.encode(
            chunks,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        chunk_embeddings = np.asarray(
            chunk_embeddings,
            dtype=np.float32,
        )

        if (
            chunk_embeddings.ndim != 2
            or chunk_embeddings.shape[0] == 0
        ):
            return chunks[:k]

        # Normalize embeddings so inner product behaves
        # like cosine similarity.
        faiss.normalize_L2(chunk_embeddings)

        # Build FAISS index.
        index = faiss.IndexFlatIP(
            chunk_embeddings.shape[1]
        )

        index.add(chunk_embeddings)

        # Create embedding for the job description.
        query_embedding = embed_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        faiss.normalize_L2(query_embedding)

        number_to_return = min(
            k,
            len(chunks),
        )

        # Retrieve the most relevant resume chunks.
        _, indices = index.search(
            query_embedding,
            number_to_return,
        )

        relevant_chunks = [
            chunks[index_value]
            for index_value in indices[0]
            if 0 <= index_value < len(chunks)
        ]

        return relevant_chunks

    except Exception as exc:
        logger.exception(
            "FAISS retrieval failed: %s",
            exc,
        )

        # Retrieval failure should not crash ResumeIQ.
        return chunks[:k]


# ---------------------------------------------------------
# Text Truncation
# ---------------------------------------------------------

def _truncate(
    text: str,
    max_chars: int,
) -> str:
    """
    Prevent excessively large resume or job-description
    text from being sent to the LLM.
    """

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n...[TRUNCATED]"
    )


# ---------------------------------------------------------
# Build Grounded RAG Context
# ---------------------------------------------------------

def build_context_block(
    match_result: MatchResult,
    jd_text: str,
    resume_text: str,
) -> ContextBlock:
    """
    Build the verified context supplied to the AI Advisor.

    The context contains:
    - ATS scores
    - matched skills
    - missing skills
    - job description
    - parsed resume text
    - FAISS-retrieved resume chunks
    """

    safe_jd_text = _truncate(
        jd_text,
        MAX_RAW_TEXT_CHARS,
    )

    safe_resume_text = _truncate(
        resume_text,
        MAX_RAW_TEXT_CHARS,
    )

    # Chunk the parsed resume text.
    resume_chunks = chunk_text(
        safe_resume_text
    )

    # Retrieve resume chunks relevant to the JD.
    relevant_chunks = retrieve_relevant_chunks(
        safe_jd_text,
        resume_chunks,
    )

    return ContextBlock(
        filename=match_result.filename,

        final_score=match_result.final_score,
        semantic_score=match_result.semantic_score,
        keyword_score=match_result.keyword_score,

        matched_skills=list(
            match_result.matched_skills
        ),

        missing_skills=list(
            match_result.missing_skills
        ),

        jd_text=safe_jd_text,
        resume_text=safe_resume_text,

        resume_chunks=relevant_chunks,
    )
# ---------------------------------------------------------
# AI Advisor Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are an AI Career Advisor and ATS Analyst for ResumeIQ.

ROLE

Your role is to analyze verified candidate evidence and provide
grounded career guidance for the supplied job description.

You will receive verified information from:

- Candidate resume
- Job description
- ATS match score
- Semantic score
- Keyword score
- Matched skills
- Missing skills
- RAG-retrieved resume chunks


GROUNDING RULES

1. Never invent candidate experience.

2. Never invent candidate projects.

3. Never invent employers or companies.

4. Never claim that the candidate possesses a skill unless the
   supplied evidence supports that claim.

5. Skills under MISSING SKILLS may only be discussed as gaps,
   learning targets, or areas for future improvement.

6. Base candidate-specific statements only on the supplied
   resume, ATS results, job description, and RAG context.

7. Never claim that a MISSING SKILL already exists on the resume.

8. Generate between 3 and 5 interview questions.

9. For every interview question, "related_skill" must contain
   EXACTLY ONE skill copied verbatim from either MATCHED SKILLS
   or MISSING SKILLS.

10. Never combine multiple skills inside "related_skill".

11. Never create a new skill label for "related_skill".

12. Every "skill" value inside "missing_skill_gaps" must be
    copied verbatim from MISSING SKILLS.

13. Never include a skill in "missing_skill_gaps" unless that
    skill appears in MISSING SKILLS.

14. Generate EXACTLY 3 resume suggestions.

15. If there are no missing skills, return:
    "missing_skill_gaps": []

16. Never invent certifications, achievements, job titles,
    education, employers, projects, or years of experience.

17. Suggestions involving missing skills must clearly describe
    them as future learning or improvement targets.

18. Do not recommend adding a missing skill to the resume as an
    existing qualification until practical evidence exists.

19. Return ONLY valid JSON.

20. Do not include Markdown, code fences, introductory text,
    explanations, or commentary outside the JSON object.


REQUIRED JSON STRUCTURE

{
  "gap_analysis": {
    "summary": "...",
    "missing_skill_gaps": [
      {
        "skill": "...",
        "why_it_matters": "...",
        "how_to_address": "..."
      }
    ]
  },
  "interview_questions": [
    {
      "question": "...",
      "related_skill": "...",
      "purpose": "..."
    }
  ],
  "resume_suggestions": [
    {
      "area": "...",
      "suggestion": "...",
      "rationale": "..."
    },
    {
      "area": "...",
      "suggestion": "...",
      "rationale": "..."
    },
    {
      "area": "...",
      "suggestion": "...",
      "rationale": "..."
    }
  ]
}


REQUIRED TOP-LEVEL KEYS

The JSON object MUST contain all three keys:

- gap_analysis
- interview_questions
- resume_suggestions

Do not omit any of these keys.


FEW-SHOT EXAMPLE 1

Example verified context:

MATCHED SKILLS:
Python, SQL

MISSING SKILLS:
Docker

Example output:

{
  "gap_analysis": {
    "summary": "The candidate matches Python and SQL requirements, while Docker remains a skill gap for the target role.",
    "missing_skill_gaps": [
      {
        "skill": "Docker",
        "why_it_matters": "Docker is relevant to containerized application development and deployment for the target role.",
        "how_to_address": "Learn Docker fundamentals and practice containerizing a small Python application."
      }
    ]
  },
  "interview_questions": [
    {
      "question": "How have you used Python in a project or technical task?",
      "related_skill": "Python",
      "purpose": "Evaluate the candidate's demonstrated understanding of Python."
    },
    {
      "question": "How would you use SQL to retrieve and filter application data?",
      "related_skill": "SQL",
      "purpose": "Evaluate the candidate's understanding of SQL."
    },
    {
      "question": "How would you approach learning and using Docker for a Python application?",
      "related_skill": "Docker",
      "purpose": "Evaluate the candidate's understanding of a skill identified as a gap."
    }
  ],
  "resume_suggestions": [
    {
      "area": "Technical Skills",
      "suggestion": "Present verified Python and SQL skills clearly in the technical-skills section.",
      "rationale": "These verified skills match requirements in the target job description."
    },
    {
      "area": "Project Evidence",
      "suggestion": "Describe existing projects with concise bullets showing how verified technical skills were applied.",
      "rationale": "Concrete evidence helps recruiters connect existing experience with job requirements."
    },
    {
      "area": "Skill Development",
      "suggestion": "After gaining practical Docker experience, add truthful supporting project evidence to the resume.",
      "rationale": "Docker is currently a missing skill and should not be presented as an existing qualification."
    }
  ]
}


FEW-SHOT EXAMPLE 2

Example verified context:

MATCHED SKILLS:
Java, Git

MISSING SKILLS:
(none)

Example output:

{
  "gap_analysis": {
    "summary": "The supplied ATS evidence does not identify any missing skills for this example.",
    "missing_skill_gaps": []
  },
  "interview_questions": [
    {
      "question": "How have you applied Java in a technical project?",
      "related_skill": "Java",
      "purpose": "Evaluate the candidate's demonstrated understanding of Java."
    },
    {
      "question": "How do you use Git when managing changes in a software project?",
      "related_skill": "Git",
      "purpose": "Evaluate the candidate's understanding of version control."
    },
    {
      "question": "How would you troubleshoot a Java project while managing changes with version control?",
      "related_skill": "Java",
      "purpose": "Evaluate problem-solving using a verified technical skill."
    }
  ],
  "resume_suggestions": [
    {
      "area": "Technical Skills",
      "suggestion": "Keep verified technical skills clearly organized and easy to identify.",
      "rationale": "Clear presentation helps recruiters identify relevant qualifications."
    },
    {
      "area": "Project Evidence",
      "suggestion": "Use concise project bullets that show how existing verified skills were applied.",
      "rationale": "Project evidence provides context for technical skills."
    },
    {
      "area": "Resume Clarity",
      "suggestion": "Keep resume descriptions concise and focused on evidence relevant to the target role.",
      "rationale": "Relevant evidence improves resume clarity and ATS readability."
    }
  ]
}


END OF FEW-SHOT EXAMPLES


Now generate a NEW response using only the ACTUAL candidate
context supplied after this prompt.

The examples above demonstrate format and reasoning only.

NEVER copy candidate facts, skills, projects, employers, or
qualifications from the examples unless the same evidence appears
in the actual candidate context.

FINAL VALIDATION RULES

- "related_skill" = exactly one skill from actual MATCHED SKILLS
  or actual MISSING SKILLS.

- "missing_skill_gaps.skill" = exactly one skill from actual
  MISSING SKILLS.

- Generate 3 to 5 interview questions.

- Generate exactly 3 resume suggestions.

- Return valid JSON only.
"""


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------

class AdviceGenerationError(RuntimeError):
    """
    Raised when AI advice cannot be generated or validated.
    """
# ---------------------------------------------------------
# Safe Fallback Helpers
# ---------------------------------------------------------

def _fallback_resume_suggestion(
    number: int,
) -> dict:
    """
    Create a safe generic resume suggestion without
    inventing candidate-specific qualifications.
    """

    fallback_options = [
        {
            "area": "Resume Relevance",
            "suggestion": (
                "Make the resume more targeted to the job description "
                "by emphasizing only qualifications genuinely supported "
                "by the resume."
            ),
            "rationale": (
                "A targeted resume makes relevant evidence easier for "
                "recruiters and ATS systems to identify."
            ),
        },
        {
            "area": "Resume Clarity",
            "suggestion": (
                "Use concise bullet points and clearly describe the "
                "responsibilities, projects, and outcomes already "
                "supported by the resume."
            ),
            "rationale": (
                "Clear descriptions make existing qualifications "
                "easier to evaluate."
            ),
        },
        {
            "area": "Skill Presentation",
            "suggestion": (
                "Organize verified technical skills into a clear skills "
                "section and avoid adding unsupported skills."
            ),
            "rationale": (
                "A structured skills section improves readability while "
                "keeping the resume accurate."
            ),
        },
    ]

    index = min(
        max(number - 1, 0),
        len(fallback_options) - 1,
    )

    return fallback_options[index]


def _fallback_interview_question(
    context: ContextBlock,
    number: int,
) -> dict:
    """
    Create a grounded fallback interview question when
    the model returns fewer than three valid questions.
    """

    available_skills = (
        list(context.matched_skills)
        + list(context.missing_skills)
    )

    if not available_skills:
        raise AdviceGenerationError(
            "Cannot create grounded interview questions because "
            "the ATS pipeline returned no matched or missing skills."
        )

    skill = available_skills[
        (number - 1) % len(available_skills)
    ]

    return {
        "question": (
            f"How would you approach a task involving {skill} "
            "for this role?"
        ),
        "related_skill": skill,
        "purpose": (
            f"Evaluate the candidate's understanding of {skill} "
            "in relation to the job requirements."
        ),
    }


# ---------------------------------------------------------
# Skill Matching Helper
# ---------------------------------------------------------

def _normalize_skill_name(skill: str) -> str:
    """
    Normalize a skill name for safe comparison.
    """

    normalized = skill.strip().lower()

    normalized = re.sub(
        r"[-_/]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized


def _find_exact_ats_skill(
    generated_skill: str,
    allowed_skills: list[str],
) -> str | None:
    """
    Match a generated skill to an ATS skill after basic
    formatting normalization.

    Returns the original ATS spelling when matched.
    """

    normalized_generated = _normalize_skill_name(
        generated_skill
    )

    for allowed_skill in allowed_skills:
        if (
            _normalize_skill_name(allowed_skill)
            == normalized_generated
        ):
            return allowed_skill

    return None


# ---------------------------------------------------------
# Normalize / Validate Groq Response
# ---------------------------------------------------------

def _normalize_advice_data(
    data: dict,
    context: ContextBlock,
) -> dict:
    """
    Repair small structural problems in the Groq response
    while enforcing ATS-grounded skill references.
    """

    if not isinstance(data, dict):
        raise AdviceGenerationError(
            "Groq response was not a JSON object."
        )

    matched_skills = list(context.matched_skills)
    missing_skills = list(context.missing_skills)

    allowed_skills = (
        matched_skills
        + missing_skills
    )

    # -----------------------------------------------------
    # Gap Analysis
    # -----------------------------------------------------

    gap_analysis = data.get("gap_analysis")

    if not isinstance(gap_analysis, dict):
        gap_analysis = {}

    summary = gap_analysis.get("summary")

    if not isinstance(summary, str) or not summary.strip():
        summary = (
            "Review the verified matched and missing skills "
            "against the target job requirements."
        )

    raw_gaps = gap_analysis.get(
        "missing_skill_gaps",
        [],
    )

    if not isinstance(raw_gaps, list):
        raw_gaps = []

    valid_gaps = []

    for item in raw_gaps:
        if not isinstance(item, dict):
            continue

        skill = item.get("skill")
        why_it_matters = item.get("why_it_matters")
        how_to_address = item.get("how_to_address")

        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                skill,
                why_it_matters,
                how_to_address,
            )
        ):
            continue

        # Gap skills MUST come from ATS missing skills.
        verified_skill = _find_exact_ats_skill(
            skill,
            missing_skills,
        )

        if verified_skill is None:
            logger.warning(
                "Removed unsupported gap skill from AI output: %s",
                skill,
            )
            continue

        valid_gaps.append(
            {
                "skill": verified_skill,
                "why_it_matters": why_it_matters.strip(),
                "how_to_address": how_to_address.strip(),
            }
        )

    data["gap_analysis"] = {
        "summary": summary.strip(),
        "missing_skill_gaps": valid_gaps,
    }

    # -----------------------------------------------------
    # Interview Questions
    # -----------------------------------------------------

    raw_questions = data.get(
        "interview_questions",
        [],
    )

    if not isinstance(raw_questions, list):
        raw_questions = []

    valid_questions = []

    for item in raw_questions:
        if not isinstance(item, dict):
            continue

        question = item.get("question")
        related_skill = item.get("related_skill")
        purpose = item.get("purpose")

        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                question,
                related_skill,
                purpose,
            )
        ):
            continue

        # related_skill MUST be one exact ATS skill.
        verified_skill = _find_exact_ats_skill(
            related_skill,
            allowed_skills,
        )

        if verified_skill is None:
            logger.warning(
                "Removed interview question with unsupported "
                "skill reference: %s",
                related_skill,
            )
            continue

        valid_questions.append(
            {
                "question": question.strip(),
                "related_skill": verified_skill,
                "purpose": purpose.strip(),
            }
        )

    # Maximum 5 questions.
    valid_questions = valid_questions[:5]

    # Minimum 3 questions.
    while len(valid_questions) < 3:
        valid_questions.append(
            _fallback_interview_question(
                context,
                len(valid_questions) + 1,
            )
        )

    data["interview_questions"] = valid_questions

    # -----------------------------------------------------
    # Resume Suggestions
    # -----------------------------------------------------

    raw_suggestions = data.get(
        "resume_suggestions",
        [],
    )

    if not isinstance(raw_suggestions, list):
        raw_suggestions = []

    valid_suggestions = []

    for item in raw_suggestions:
        if not isinstance(item, dict):
            continue

        area = item.get("area")
        suggestion = item.get("suggestion")
        rationale = item.get("rationale")

        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                area,
                suggestion,
                rationale,
            )
        ):
            continue

        valid_suggestions.append(
            {
                "area": area.strip(),
                "suggestion": suggestion.strip(),
                "rationale": rationale.strip(),
            }
        )

    # Exactly 3 suggestions.
    valid_suggestions = valid_suggestions[:3]

    while len(valid_suggestions) < 3:
        valid_suggestions.append(
            _fallback_resume_suggestion(
                len(valid_suggestions) + 1
            )
        )

    data["resume_suggestions"] = valid_suggestions

    return data


# ---------------------------------------------------------
# Generate AI Advice
# ---------------------------------------------------------

def generate_advice(
    context: ContextBlock,
    model: str = LLM_MODEL,
) -> AdviceOutput:
    """
    Generate structured, grounded career advice using Groq.
    """

    prompt = f"""
{SYSTEM_PROMPT}

ACTUAL VERIFIED CANDIDATE CONTEXT
---------------------------------

{context.as_prompt_text()}

Return ONLY the required valid JSON object.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ResumeIQ's grounded AI Career Advisor. "
                        "Follow the supplied role, grounding rules, "
                        "few-shot examples, and JSON schema. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            response_format={
                "type": "json_object"
            },
            max_tokens=4000,
        )

    except Exception as exc:
        logger.exception(
            "Groq API request failed: %s",
            exc,
        )

        raise AdviceGenerationError(
            "The AI Career Advisor could not generate advice "
            "right now. Please try again."
        ) from exc

    try:
        raw_content = (
            response
            .choices[0]
            .message
            .content
        )

        if not raw_content:
            raise AdviceGenerationError(
                "Groq returned an empty response."
            )

        data = json.loads(
            raw_content.strip()
        )

    except json.JSONDecodeError as exc:
        logger.exception(
            "Groq returned invalid JSON."
        )

        raise AdviceGenerationError(
            "The AI response could not be validated. "
            "Please generate the advice again."
        ) from exc

    except (AttributeError, IndexError) as exc:
        logger.exception(
            "Unexpected Groq response structure."
        )

        raise AdviceGenerationError(
            "The AI service returned an unexpected response."
        ) from exc

    # Enforce structural and groundedness requirements.
    data = _normalize_advice_data(
        data,
        context,
    )

    try:
        advice = AdviceOutput(
            **data
        )

    except ValidationError as exc:
        logger.error(
            "AI advice validation failed: %s",
            exc,
        )

        raise AdviceGenerationError(
            "The AI response failed structured-output validation."
        ) from exc

    return advice


# ---------------------------------------------------------
# Follow-up Chat
# ---------------------------------------------------------

def ask_followup(
    context: ContextBlock,
    question: str,
) -> str:
    """
    Answer follow-up questions using verified ResumeIQ context.

    The assistant may explain matched skills, missing skills,
    job requirements and practical ways to improve a skill.

    Missing skills must never be presented as skills the
    candidate already possesses.
    """

    if not question or not question.strip():
        return "Please enter a question."

    followup_prompt = f"""
You are the ResumeIQ AI Career Advisor.

VERIFIED CANDIDATE CONTEXT
--------------------------

{context.as_prompt_text()}

USER QUESTION
-------------

{question.strip()}

FOLLOW-UP RULES

1. Answer using only the supplied resume, job description,
   ATS results, matched skills, missing skills and RAG context.

2. Never invent candidate experience, projects, employers,
   certifications, achievements, education or qualifications.

3. MATCHED SKILLS are skills verified by the ATS pipeline.

4. MISSING SKILLS are gaps for the target role.
   Never claim that the candidate already possesses them.

5. If the user asks how to improve or learn a MISSING SKILL,
   you MAY provide practical learning steps, study topics,
   exercises or project ideas for developing that skill.

6. Learning guidance is future guidance. Never describe it
   as something the candidate has already completed.

7. If the user asks about a job-description requirement,
   explain it using the supplied job description.

8. If the user asks about existing candidate experience,
   answer only when resume or ATS evidence supports it.

9. Only say:
   "This information is not mentioned in the resume or job description."
   when the requested topic truly does not appear in the
   supplied context.

10. Keep the response concise, practical and relevant.

11. Format the answer using simple Markdown headings and bullet points.

12. Do NOT use Markdown tables.

13. Do NOT use HTML tags such as <br>, <div>, <p>, or <table>.

14. For learning or improvement questions, prefer a simple numbered
    learning path with short bullet points.

Answer the user's question directly.
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ResumeIQ's grounded AI Career Advisor. "
                        "Use verified context for candidate-specific "
                        "claims. You may provide practical future "
                        "learning guidance for ATS missing skills."
                    ),
                },
                {
                    "role": "user",
                    "content": followup_prompt,
                },
            ],
            temperature=0.1,
            max_tokens=700,
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            return (
                "The AI Advisor returned an empty response."
            )

        return content.strip()

    except Exception as exc:
        logger.exception(
            "Follow-up question failed: %s",
            exc,
        )

        # Do not expose raw API errors in the UI.
        return (
            "The AI Advisor could not process the follow-up "
            "question right now. Please try again."
        )
# ---------------------------------------------------------
# Groundedness Report
# ---------------------------------------------------------

class GroundednessReport(BaseModel):
    """
    Result of checking structured skill references
    produced by the AI Career Advisor.
    """

    is_grounded: bool

    checked_skill_mentions: list[str]

    unverified_skills: list[str]


# ---------------------------------------------------------
# Groundedness Check
# ---------------------------------------------------------

def check_groundedness(
    context: ContextBlock,
    advice: AdviceOutput,
) -> GroundednessReport:
    """
    Verify structured skill references produced by the AI
    against skills supplied by the ATS pipeline.

    Allowed skill references are:
    - matched skills
    - missing skills discussed as gaps

    A missing skill must never be treated as an existing
    candidate qualification.
    """

    allowed_skills = (
        list(context.matched_skills)
        + list(context.missing_skills)
    )

    mentioned_skills: list[str] = []

    # Check gap-analysis skill references.
    for gap in advice.gap_analysis.missing_skill_gaps:
        skill = gap.skill.strip()

        if skill:
            mentioned_skills.append(skill)

    # Check interview-question skill references.
    for interview_question in advice.interview_questions:
        skill = interview_question.related_skill.strip()

        if skill:
            mentioned_skills.append(skill)

    unverified_skills: list[str] = []

    for skill in mentioned_skills:
        verified_skill = _find_exact_ats_skill(
            skill,
            allowed_skills,
        )

        if verified_skill is None:
            unverified_skills.append(skill)

    unique_mentions = sorted(
        set(mentioned_skills),
        key=str.lower,
    )

    unique_unverified = sorted(
        set(unverified_skills),
        key=str.lower,
    )

    if unique_unverified:
        logger.warning(
            "Unverified AI skill references detected: %s",
            unique_unverified,
        )

    return GroundednessReport(
        is_grounded=(
            len(unique_unverified) == 0
        ),
        checked_skill_mentions=unique_mentions,
        unverified_skills=unique_unverified,
    )


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    try:
        from src.parser import parse_document
        from src.preprocess import clean_text
        from src.scoring import score_resume

    except ImportError as exc:
        logger.error(
            "Import error: %s",
            exc,
        )

        raise SystemExit(
            "Please ensure all src modules are available."
        )


    # -----------------------------------------------------
    # Sample Job Description
    # -----------------------------------------------------

    jd_raw = """
    We are looking for a Python Developer with experience
    in Python, FastAPI, Docker, AWS, PostgreSQL, Git,
    CI/CD, Kubernetes and REST APIs.
    """


    # -----------------------------------------------------
    # Sample Resume
    # -----------------------------------------------------

    sample_resume_path = (
        "data/sample_resumes/john_doe.docx"
    )

    try:
        with open(
            sample_resume_path,
            "rb",
        ) as file:

            parsed = parse_document(
                file.read(),
                "john_doe.docx",
            )

    except FileNotFoundError:
        raise SystemExit(
            f"Sample resume not found: "
            f"{sample_resume_path}"
        )


    if not parsed.is_usable:
        raise SystemExit(
            parsed.error_message
        )


    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    jd_clean = clean_text(
        jd_raw
    )

    resume_clean = clean_text(
        parsed.raw_text
    )


    # -----------------------------------------------------
    # ATS Scoring
    # -----------------------------------------------------

    logger.info(
        "Scoring resume..."
    )

    result = score_resume(
        jd_clean,
        resume_clean,
        parsed.filename,
    )

    print("\n" + "=" * 50)

    print(
        f"FIT SCORE: "
        f"{result.final_score:.2f}/100"
    )

    print("=" * 50)

    print(
        "Matched Skills:",
        result.matched_skills,
    )

    print(
        "Missing Skills:",
        result.missing_skills,
    )


    # -----------------------------------------------------
    # Build RAG Context
    # -----------------------------------------------------

    logger.info(
        "Building RAG context..."
    )

    context = build_context_block(
        result,
        jd_clean,
        resume_clean,
    )


    # -----------------------------------------------------
    # Generate AI Advice
    # -----------------------------------------------------

    logger.info(
        "Generating AI advice..."
    )

    try:
        advice = generate_advice(
            context
        )

    except AdviceGenerationError as exc:
        print(
            f"\nAI Advisor Error:\n{exc}"
        )

        raise SystemExit(1)


    print(
        "\nAI CAREER ADVICE"
    )

    print(
        json.dumps(
            advice.model_dump(),
            indent=2,
        )
    )


    # -----------------------------------------------------
    # Groundedness Verification
    # -----------------------------------------------------

    logger.info(
        "Checking groundedness..."
    )

    report = check_groundedness(
        context,
        advice,
    )

    print(
        "\nGROUNDEDNESS REPORT"
    )

    print(
        json.dumps(
            report.model_dump(),
            indent=2,
        )
    )


    # -----------------------------------------------------
    # Follow-up Chat Test
    # -----------------------------------------------------

    print(
        "\nFOLLOW-UP TEST"
    )

    question = (
        "How can I improve my AWS skills for this role?"
    )

    print(
        "Question:",
        question,
    )

    print(
        "Answer:",
        ask_followup(
            context,
            question,
        ),
    )