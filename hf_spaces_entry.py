"""
app.py

Entry point for Hugging Face Spaces deployment.

HF Spaces expects a root-level app.py for Streamlit Spaces. Rather than
duplicating the dashboard code, this just imports and runs the real app
from app/streamlit_app.py -- keeping one source of truth for the UI.
"""

import runpy

runpy.run_path("app/streamlit_app.py", run_name="__main__")