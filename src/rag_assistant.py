"""
src/rag_assistant.py

RAG layer for the AI Career Advisor. Takes the existing scoring output
(src/scoring.py's MatchResult) plus the raw JD/resume text, and packages
them into a grounding "context block" that gets handed to a local LLM
(via Ollama) to generate a gap analysis, interview questions, and resume
rewrite suggestions.

Design principle: the LLM only ever sees evidence that already came out
of scoring.py / skills_data.py -- matched skills, missing skills, and the
raw JD/resume text. It is never asked to invent new skills, only to
reason over what's already been retrieved. Phase 3's groundedness check
verifies the LLM's output stayed within this evidence.
"""

from __future__ import annotations

import json
import logging

import ollama
from pydantic import BaseModel

from src.scoring import MatchResult

logger = logging.getLogger(__name__)

# Rough char budget for raw text embedded in the prompt. Resumes/JDs in
# this project are short (a page or two of cleaned text), so this is a
# safety net against pathological inputs, not a real constraint in
# practice -- see the chunking note in build_context_block() below.
MAX_RAW_TEXT_CHARS = 6000


class ContextBlock(BaseModel):
    """
    Grounding context passed to the LLM for one resume/JD pair.

    Every field here was already produced by scoring.py / skills_data.py
    -- nothing in this object is LLM-generated. This is the "evidence"
    that the Phase 3 groundedness check will check the LLM's response
    against, so keep it as the single source of truth for what the LLM
    is and isn't allowed to talk about.
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
        Render this context block as the plain-text block embedded in the
        LLM prompt. Kept separate from the data model so the prompt
        format can be tuned without touching the underlying structure.
        """
        matched = ", ".join(self.matched_skills) if self.matched_skills else "(none)"
        missing = ", ".join(self.missing_skills) if self.missing_skills else "(none)"

        return f"""CANDIDATE: {self.filename}
MATCH SCORE: {self.final_score}/100 (semantic: {self.semantic_score}/100, keyword: {self.keyword_score}/100)

MATCHED SKILLS (confirmed present in both the JD and the resume):
{matched}

MISSING SKILLS (required by the JD, not found in the resume):
{missing}

JOB DESCRIPTION:
{self.jd_text}

RESUME:
{self.resume_text}"""


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def build_context_block(
    match_result: MatchResult,
    jd_text: str,
    resume_text: str,
) -> ContextBlock:
    """
    Assemble the grounding context for one resume/JD pair.

    match_result should come straight out of scoring.score_resume() --
    this function doesn't recompute anything, it repackages what
    scoring.py and skills_data.py already extracted (matched/missing
    skills, scores) alongside the raw text, so the LLM has enough detail
    to write specific advice rather than generic filler.

    jd_text / resume_text should be the same *cleaned* text that was
    passed into score_resume() for this match_result, so what the LLM
    sees is consistent with what was actually scored.

    Chunking note (relevant if this project scales up):
    At the current scale -- single-page resumes, JDs of a few hundred
    to low-thousands of words -- whole documents fit comfortably inside
    an 8B-class local model's context window, so this function passes
    them through directly with a generous truncation safety net. If this
    were extended to longer documents (multi-page CVs, lengthy JDs), the
    right place to add chunking is here: split jd_text/resume_text into
    sections (e.g. by resume headers like "Experience" / "Skills"),
    embed each chunk with embed_batch() from embeddings.py, and retrieve
    only the chunks most relevant to the missing_skills terms -- rather
    than passing full documents through on every call.
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


# ---------------------------------------------------------------------------
# Phase 2 -- generation layer
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "llama3.1:8b"

# Grounding is enforced two ways: (1) the system prompt tells the model to
# only use the skills it was handed, and (2) Phase 3's groundedness check
# verifies that constraint held after the fact. Neither one alone is
# reliable -- LLMs don't perfectly follow instructions -- so treat this
# prompt as the first line of defense, not the only one.
SYSTEM_PROMPT = """You are a career advisor assistant helping a job candidate understand how their resume compares to a job description.

You are given:
- A match score breakdown
- MATCHED SKILLS: skills confirmed present in both the JD and the resume
- MISSING SKILLS: skills required by the JD but not found in the resume
- The full job description text
- The full resume text

STRICT RULES:
1. Only ever refer to skills that appear in the MATCHED SKILLS or MISSING SKILLS lists. Do not invent, assume, or mention any other skill, tool, or technology -- even if it seems related or commonly paired with what's listed.
2. Base every specific claim about the candidate's experience on text that actually appears in the RESUME section. Do not invent projects, companies, employers, or experience that isn't there.
3. Respond with ONLY a single JSON object. No markdown code fences, no preamble, no explanation outside the JSON.

Return JSON in exactly this shape:
{
  "gap_analysis": {
    "summary": "2-3 sentence overview of how well the candidate matches the role",
    "missing_skill_gaps": [
      {"skill": "...", "why_it_matters": "...", "how_to_address": "..."}
    ]
  },
  "interview_questions": [
    {"question": "...", "related_skill": "...", "purpose": "..."}
  ],
  "resume_suggestions": [
    {"area": "...", "suggestion": "...", "rationale": "..."}
  ]
}

If MISSING SKILLS is empty, missing_skill_gaps should be an empty list. Generate 3-5 interview_questions and 2-4 resume_suggestions, all grounded in the actual resume content provided."""


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama isn't reachable (e.g. the server isn't running)."""


class AdviceGenerationError(RuntimeError):
    """Raised when the model's response isn't valid JSON, or doesn't match
    the expected advice shape."""


def generate_advice(context: ContextBlock, model: str = OLLAMA_MODEL) -> dict:
    """
    Sends the grounding context to the local LLM (via Ollama) and returns
    structured advice: gap analysis, interview questions, and resume
    rewrite suggestions.

    Raises:
        OllamaConnectionError: if Ollama isn't running / unreachable, or
            the model isn't pulled.
        AdviceGenerationError: if the model's response isn't valid JSON,
            or is valid JSON but missing expected keys.
    """
    user_prompt = context.as_prompt_text()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",  # constrains Ollama to emit syntactically valid JSON
            options={"temperature": 0.2},  # low temp: grounded and consistent, not creative
        )
    except Exception as exc:
        # The ollama client raises different exception types depending on
        # the failure (connection refused, model not found, etc.) --
        # normalize all of them into one clear, actionable error.
        raise OllamaConnectionError(
            f"Could not get a response from Ollama (model='{model}'). "
            f"Is Ollama running and is the model pulled "
            f"(`ollama pull {model}`)? Original error: {exc}"
        ) from exc

    raw_content = response["message"]["content"]

    try:
        advice = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise AdviceGenerationError(
            f"Model response wasn't valid JSON. Raw response:\n{raw_content}"
        ) from exc

    expected_keys = {"gap_analysis", "interview_questions", "resume_suggestions"}
    missing_keys = expected_keys - advice.keys()
    if missing_keys:
        raise AdviceGenerationError(
            f"Model response is missing expected keys: {missing_keys}. "
            f"Raw response:\n{raw_content}"
        )

    return advice


# ---------------------------------------------------------------------------
# Phase 3 -- groundedness check (hallucination guardrail)
# ---------------------------------------------------------------------------

class GroundednessReport(BaseModel):
    """
    Result of checking one generate_advice() response against the
    ContextBlock it was generated from.
    """

    is_grounded: bool
    checked_skill_mentions: list[str]
    unverified_skills: list[str]  # skills claimed that aren't in matched/missing lists


def check_groundedness(context: ContextBlock, advice: dict) -> GroundednessReport:
    """
    Verifies that every skill the LLM explicitly named as a "gap" or as
    "related_skill" for an interview question actually appears in the
    matched_skills / missing_skills lists that were fed into the prompt
    as evidence.

    Scope note: this intentionally checks only two structured fields --
    missing_skill_gaps[].skill and interview_questions[].related_skill --
    because those are the fields where the prompt explicitly promises
    "only skills from MATCHED/MISSING SKILLS". resume_suggestions is
    free-form advice about the resume's existing content and is allowed
    to reference anything actually present in the resume text, not just
    the matched/missing skill vocabulary, so it isn't checked here.

    Any skill mentioned outside the allowed set is logged as a possible
    hallucination -- this is the guardrail Phase 3 calls for. It doesn't
    block or rewrite the response; that decision (retry, discard, warn
    the user) is left to the caller (API/UI layer in later phases).
    """
    allowed_skills = {
        s.strip().lower() for s in (context.matched_skills + context.missing_skills)
    }

    mentioned: list[str] = []

    gap_analysis = advice.get("gap_analysis", {})
    for gap in gap_analysis.get("missing_skill_gaps", []):
        skill = gap.get("skill")
        if skill:
            mentioned.append(skill)

    for question in advice.get("interview_questions", []):
        skill = question.get("related_skill")
        if skill:
            mentioned.append(skill)

    unverified = [
        skill for skill in mentioned if skill.strip().lower() not in allowed_skills
    ]

    if unverified:
        logger.warning(
            "Groundedness check flagged possible hallucinated skill(s) for "
            "%s: %s (allowed skills were: %s)",
            context.filename,
            unverified,
            sorted(allowed_skills),
        )

    return GroundednessReport(
        is_grounded=len(unverified) == 0,
        checked_skill_mentions=mentioned,
        unverified_skills=unverified,
    )


if __name__ == "__main__":
    # Manual test per Phase 2 checklist -- run with: python -m src.rag_assistant
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.parser import parse_document
    from src.preprocess import clean_text
    from src.scoring import score_resume

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

    print("Generating advice via Ollama -- this may take a bit on first run...\n")

    try:
        advice = generate_advice(context)
    except (OllamaConnectionError, AdviceGenerationError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    print(json.dumps(advice, indent=2))

    print("\n--- Groundedness check ---")
    report = check_groundedness(context, advice)
    print(f"is_grounded: {report.is_grounded}")
    print(f"checked_skill_mentions: {report.checked_skill_mentions}")
    print(f"unverified_skills: {report.unverified_skills}")
