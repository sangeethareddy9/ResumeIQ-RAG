"""
src/rag_assistant.py

ResumeIQ - AI Resume Screening System

This module is responsible for:

1. Building the Retrieval Augmented Generation (RAG) context.
2. Sending the context to Groq Llama.
3. Generating career advice.
4. Performing groundedness verification.

The AI only reasons over evidence extracted during ATS scoring.
It is never allowed to invent candidate skills.
"""

from __future__ import annotations

import json
import logging
import os

from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel

from src.scoring import MatchResult

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

LLM_MODEL = "llama-3.3-70b-versatile"

MAX_RAW_TEXT_CHARS = 6000


# ---------------------------------------------------------
# Context Block
# ---------------------------------------------------------

class ContextBlock(BaseModel):
    """
    Represents all evidence supplied to Groq Llama.

    Everything inside this object has already been verified
    by the ATS scoring pipeline.
    """

    filename: str
    final_score: float
    semantic_score: float
    keyword_score: float

    matched_skills: list[str]
    missing_skills: list[str]

    jd_text: str
    resume_text: str

    def as_prompt_text(self) -> str:
        """
        Converts the context into a prompt-ready string.
        """

        matched = ", ".join(self.matched_skills)

        if not matched:
            matched = "(none)"

        missing = ", ".join(self.missing_skills)

        if not missing:
            missing = "(none)"

        return f"""
Candidate:
{self.filename}

Overall Match:
{self.final_score:.2f}%

Semantic Score:
{self.semantic_score:.2f}%

Keyword Score:
{self.keyword_score:.2f}%

Matched Skills:
{matched}

Missing Skills:
{missing}

==========================
JOB DESCRIPTION
==========================

{self.jd_text}

==========================
RESUME
==========================

{self.resume_text}
"""


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    """
    Prevent extremely large prompts.
    """

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[TRUNCATED]"

# ---------------------------------------------------------
# Build RAG Context
# ---------------------------------------------------------

def build_context_block(
    match_result: MatchResult,
    jd_text: str,
    resume_text: str,
) -> ContextBlock:
    """
    Creates the grounding context passed to Groq Llama.

    The model only receives:
    - ATS Scores
    - Matched Skills
    - Missing Skills
    - Resume
    - Job Description

    No additional knowledge is injected.
    """

    return ContextBlock(
        filename=match_result.filename,
        final_score=match_result.final_score,
        semantic_score=match_result.semantic_score,
        keyword_score=match_result.keyword_score,
        matched_skills=match_result.matched_skills,
        missing_skills=match_result.missing_skills,
        jd_text=_truncate(jd_text, MAX_RAW_TEXT_CHARS),
        resume_text=_truncate(resume_text, MAX_RAW_TEXT_CHARS),
    )


# ---------------------------------------------------------
# Groq Llama Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert AI Career Advisor.

You will receive:

• Resume
• Job Description
• ATS Score
• Semantic Score
• Keyword Score
• Matched Skills
• Missing Skills

IMPORTANT RULES

1. Never invent candidate experience.

2. Never invent projects.

3. Never invent companies.

4. Never mention skills that are NOT present inside either:

Matched Skills

OR

Missing Skills

5. Respond ONLY in JSON.

Return this structure exactly:

{
  "gap_analysis": {
    "summary": "...",
    "missing_skill_gaps": [
      {
        "skill":"...",
        "why_it_matters":"...",
        "how_to_address":"..."
      }
    ]
  },

  "interview_questions":[
    {
      "question":"...",
      "related_skill":"...",
      "purpose":"..."
    }
  ],

  "resume_suggestions":[
    {
      "area":"...",
      "suggestion":"...",
      "rationale":"..."
    }
  ]
}

Generate

• 3-5 Interview Questions

• 2-4 Resume Suggestions

If there are no missing skills,
return an empty missing_skill_gaps list.
"""


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------

class AdviceGenerationError(RuntimeError):
    """
    Raised when Groq Llama produces
    invalid or incomplete JSON.
    """
    
# ---------------------------------------------------------
# Generate AI Advice using Groq Llama
# ---------------------------------------------------------

def generate_advice(context: ContextBlock, model: str = LLM_MODEL) -> dict:
    """
    Generates AI career advice using Groq Llama.
    Returns the same JSON structure expected by the rest of the project.
    """

    prompt = f"""
{SYSTEM_PROMPT}

{context.as_prompt_text()}

Return ONLY valid JSON.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI Career Advisor.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content.strip()

    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate response from Groq: {exc}"
        ) from exc

    try:
        advice = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise AdviceGenerationError(
            f"Groq returned invalid JSON:\n{raw_content}"
        ) from exc

    expected_keys = {
        "gap_analysis",
        "interview_questions",
        "resume_suggestions",
    }

    missing = expected_keys - advice.keys()

    if missing:
        raise AdviceGenerationError(
            f"Groq response missing keys: {missing}"
        )

    return advice
# ---------------------------------------------------------
# Groundedness Check
# ---------------------------------------------------------

class GroundednessReport(BaseModel):
    """
    Verifies that Groq Llama only talks about skills
    present in the retrieved RAG context.
    """

    is_grounded: bool
    checked_skill_mentions: list[str]
    unverified_skills: list[str]


def check_groundedness(
    context: ContextBlock,
    advice: dict,
) -> GroundednessReport:
    """
    Ensures Groq Llama only references
    matched or missing skills.
    """

    allowed_skills = {
        skill.strip().lower()
        for skill in (
            context.matched_skills +
            context.missing_skills
        )
    }

    mentioned = []

    # ----------------------------
    # Gap Analysis
    # ----------------------------

    gap_analysis = advice.get(
        "gap_analysis",
        {}
    )

    for gap in gap_analysis.get(
        "missing_skill_gaps",
        []
    ):

        skill = gap.get("skill")

        if skill:
            mentioned.append(skill)

    # ----------------------------
    # Interview Questions
    # ----------------------------

    for question in advice.get(
        "interview_questions",
        []
    ):

        skill = question.get(
            "related_skill"
        )

        if skill:
            mentioned.append(skill)

    # ----------------------------
    # Verify
    # ----------------------------

    unverified = []

    for skill in mentioned:

        if skill.strip().lower() not in allowed_skills:
            unverified.append(skill)

    if unverified:

        logger.warning(
            "Possible hallucinated skills: %s",
            unverified,
        )

    return GroundednessReport(
        is_grounded=len(unverified) == 0,
        checked_skill_mentions=mentioned,
        unverified_skills=unverified,
    )


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    from src.parser import parse_document
    from src.preprocess import clean_text
    from src.scoring import score_resume

    jd_raw = """
We are looking for a Python Developer
with experience in

Python
FastAPI
Docker
AWS
PostgreSQL
Git
CI/CD
"""

    with open(
        "data/sample_resumes/john_doe.docx",
        "rb",
    ) as file:

        parsed = parse_document(
            file.read(),
            "john_doe.docx",
        )

    if not parsed.is_usable:

        raise SystemExit(
            parsed.error_message
        )

    jd_clean = clean_text(jd_raw)

    resume_clean = clean_text(
        parsed.raw_text
    )

    result = score_resume(
        jd_clean,
        resume_clean,
        parsed.filename,
    )

    context = build_context_block(
        result,
        jd_clean,
        resume_clean,
    )

    print("Generating advice via Groq...\n")

    try:

        advice = generate_advice(context)

    except AdviceGenerationError as exc:

        print(exc)

        raise SystemExit(1)

    print(
        json.dumps(
            advice,
            indent=2,
        )
    )

    report = check_groundedness(
        context,
        advice,
    )

    print("\nGroundedness Report")

    print("----------------------")

    print(
        "Grounded:",
        report.is_grounded,
    )

    print(
        "Mentioned Skills:",
        report.checked_skill_mentions,
    )

    print(
        "Unverified Skills:",
        report.unverified_skills,
    )