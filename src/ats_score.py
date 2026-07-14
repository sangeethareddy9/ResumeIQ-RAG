"""
ats_score.py

Calculates an ATS (Applicant Tracking System) score for a resume.

The ATS score is based on:
1. Contact Information
2. Education
3. Technical Skills
4. Projects
5. Experience
6. Keyword Match with Job Description

Final Score = 100
"""

import re


def calculate_ats_score(resume_text: str, jd_skills: set, resume_skills: set) -> dict:
    """
    Calculate ATS score.

    Returns:
        {
            "score": 85,
            "breakdown": {...}
        }
    """

    score = 0
    breakdown = {}

    text = resume_text.lower()

    # -------------------------
    # Contact Information
    # -------------------------

    contact_score = 0

    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text):
        contact_score += 5

    if re.search(r"\b\d{10}\b", resume_text):
        contact_score += 5

    breakdown["Contact"] = contact_score
    score += contact_score

    # -------------------------
    # Education
    # -------------------------

    education_score = 0

    education_keywords = [
        "b.tech",
        "btech",
        "mca",
        "b.sc",
        "bsc",
        "degree",
        "university",
    ]

    if any(word in text for word in education_keywords):
        education_score = 15

    breakdown["Education"] = education_score
    score += education_score

    # -------------------------
    # Projects
    # -------------------------

    project_score = 0

    if "project" in text:
        project_score = 15

    breakdown["Projects"] = project_score
    score += project_score

    # -------------------------
    # Experience
    # -------------------------

    experience_score = 0

    experience_keywords = [
        "experience",
        "internship",
        "developer",
        "engineer",
        "trainee",
    ]

    if any(word in text for word in experience_keywords):
        experience_score = 15

    breakdown["Experience"] = experience_score
    score += experience_score

    # -------------------------
    # Skills
    # -------------------------

    skills_score = min(len(resume_skills), 10)

    skills_score *= 2

    breakdown["Skills"] = skills_score
    score += skills_score

    # -------------------------
    # JD Keyword Match
    # -------------------------

    keyword_score = 0

    if jd_skills:
        keyword_score = int(
            (len(jd_skills & resume_skills) / len(jd_skills)) * 30
        )

    breakdown["Keyword Match"] = keyword_score
    score += keyword_score

    # Maximum = 100

    score = min(score, 100)

    return {
        "score": score,
        "breakdown": breakdown,
    }