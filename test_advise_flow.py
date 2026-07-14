"""
Manual end-to-end test for the /screen -> /advise flow.
Run with the server already running (uvicorn app.fastapi_app:app --reload --port 8000),
in a separate terminal: python test_advise_flow.py
"""

import requests

BASE_URL = "http://127.0.0.1:8000"

jd_text = (
    "We are looking for a Python developer with experience in AWS, "
    "Docker, and FastAPI. Familiarity with PostgreSQL and CI/CD pipelines is a plus."
)

# --- Step 1: /screen ---
with open("data/sample_resumes/john_doe.docx", "rb") as f:
    screen_response = requests.post(
        f"{BASE_URL}/screen",
        data={"job_description": jd_text},
        files={"resumes": ("john_doe.docx", f, "application/octet-stream")},
    )

print("=== /screen response ===")
print(screen_response.status_code)
screen_data = screen_response.json()
print(screen_data)

filename = screen_data["results"][0]["filename"]

# --- Step 2: /advise ---
advise_response = requests.post(
    f"{BASE_URL}/advise",
    data={"filename": filename, "job_description": jd_text},
)

print("\n=== /advise response ===")
print(advise_response.status_code)
print(advise_response.json())
