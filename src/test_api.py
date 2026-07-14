"""
src/test_api.py

Tests the live FastAPI /screen endpoint by sending a real HTTP request
with a job description and sample resumes -- exactly how a real client
(Streamlit, a mobile app, curl, etc.) would call it.

Requires the API server to already be running:
    uvicorn app.fastapi_app:app --reload --port 8000

Run from project root (in a SEPARATE terminal from the server):
    python -m src.test_api
"""

from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000/screen"

SAMPLE_DIR = Path("data/sample_resumes")

job_description = (
    "We are hiring a Senior Software Engineer to join our backend team. "
    "Required skills: Python, FastAPI, REST API, cloud deployment (AWS), "
    "Docker, and experience with CI/CD pipelines."
)

resume_files = list(SAMPLE_DIR.glob("*.pdf")) + list(SAMPLE_DIR.glob("*.docx"))

if not resume_files:
    print(f"No resume files found in {SAMPLE_DIR.resolve()}")
else:
    # requests needs files passed as a list of (field_name, (filename, file_object, content_type))
    files = [
        ("resumes", (f.name, open(f, "rb"), "application/octet-stream"))
        for f in resume_files
    ]
    data = {"job_description": job_description}

    print(f"Sending {len(files)} resume(s) to {API_URL} ...")
    response = requests.post(API_URL, data=data, files=files)

    print(f"\nStatus code: {response.status_code}\n")

    if response.status_code == 200:
        result = response.json()
        print("=== Ranked Results ===\n")
        for r in result["results"]:
            print(f"{r['filename']}  --  {r['match_score']}% match")
            print(f"   Semantic: {r['semantic_score']}%  |  Keyword: {r['keyword_score']}%")
            print(f"   Matched: {', '.join(r['matched_skills']) or '(none)'}")
            print(f"   Missing: {', '.join(r['missing_skills']) or '(none)'}")
            print()
        if result["skipped_files"]:
            print(f"Skipped files: {result['skipped_files']}")
    else:
        print("Error response:")
        print(response.json())