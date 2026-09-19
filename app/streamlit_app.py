"""
ResumeIQ Pro - AI Resume Screening System using RAG + Groq

Author: Sangeetha Chirla
Tech: Python, Streamlit, Sentence Transformers, FAISS, Groq AI 
Description: ATS + Semantic Scoring + RAG Career Advisor with Chat
Run: streamlit run app/streamlit_app.py
"""

# ==========================================================
# Import Libraries
# ==========================================================

import sys
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow importing from src folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ==========================================================
# Import Project Modules
# ==========================================================

from src.parser import parse_document
from src.preprocess import clean_text
from src.scoring import score_resume, MatchResult
from src.rag_assistant import (
    build_context_block,
    generate_advice,
    check_groundedness,
    ask_followup,
    AdviceGenerationError,
)

# ==========================================================
# Streamlit Page Configuration
# ==========================================================

st.set_page_config(
    page_title="ResumeIQ Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# Custom CSS - Professional Violet + Black UI
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    * {font-family: 'Inter', sans-serif;}
    #MainMenu, footer, header {visibility: hidden;}
 .stApp {
        background: #0a0a0a;
        background-image: radial-gradient(at 20% 0%, #4c1d95 0px, transparent 50%),
                          radial-gradient(at 80% 0%, #7c3aed 0px, transparent 50%);
    }
 .block-container {
        background: #111; border: 1px solid #262626; padding: 2.5rem 3rem;
        border-radius: 20px; box-shadow: 0 0 40px rgba(124, 58, 237, 0.15);
    }
 .hero-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(90deg, #a855f7, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center;
    }
 .hero-subtitle {
        text-align: center; color: #a1a1aa!important; font-size: 1.05rem;
        margin-bottom: 2.5rem;
    }
 .feature-card {
        background: #18181b; padding: 1.5rem; border-radius: 16px;
        border: 1px solid #27272a; text-align: center; transition: all 0.3s ease;
    }
 .feature-card:hover {
        transform: translateY(-5px); border-color: #7c3aed;
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.3);
    }
 .feature-icon {font-size: 2.2rem;}
 .feature-title {font-weight: 600; color: #fafafa!important; margin-top: 0.5rem;}
 .feature-desc {font-size: 0.85rem; color: #a1a1aa!important;}
 .stButton>button {
        background: linear-gradient(90deg, #7c3aed, #a855f7)!important;
        color: white!important; border-radius: 10px; border: none;
        padding: 0.8rem 1.5rem; font-weight: 700; width: 100%;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.4);
    }
 .stButton>button:hover {
        transform: scale(1.02); box-shadow: 0 0 30px rgba(168, 85, 247, 0.6);
    }
    [data-testid="stFileUploader"] {
        background: #18181b!important; border: 2px dashed #3f3f46!important;
        border-radius: 12px!important;
    }
    [data-testid="stFileUploader"] button {
        background: #7c3aed!important; color: white!important;
    }
    [data-testid="stFileUploader"] * {color: #d4d4d8!important;}
    textarea {
        background: #18181b!important; color: #fafafa!important;
        border: 2px solid #3f3f46!important; border-radius: 12px!important;
    }
    textarea:focus {
        border-color: #7c3aed!important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2)!important;
    }
    [data-testid="stRadio"] label {color: #d4d4d8!important; font-weight: 500!important;}
 .stTabs [data-baseweb="tab-list"] {
        background: #18181b; border-radius: 10px; padding: 4px; border: 1px solid #27272a;
    }
 .stTabs [data-baseweb="tab"] {
        color: #a1a1aa!important; font-weight: 600; border-radius: 8px;
    }
 .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7c3aed, #a855f7);
        color: white!important; box-shadow: 0 0 15px rgba(124, 58, 237, 0.5);
    }
    [data-testid="stMetric"] {
        background: #18181b; padding: 1.2rem; border-radius: 12px;
        border: 1px solid #27272a; border-left: 4px solid #7c3aed;
    }
    [data-testid="stMetricValue"] {color: #fafafa!important; font-size: 1.8rem;}
    [data-testid="stMetricLabel"] {color: #a1a1aa!important;}
    h2, h3, h4 {color: #fafafa!important;}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Hero Section
# ==========================================================

st.markdown('<h1 class="hero-title">🚀 ResumeIQ Pro</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">AI-Powered Resume Screening & Career Advisor using Retrieval-Augmented Generation (RAG) + Groq AI </p>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
cards = [
    ("🧠", "AI Resume Screening", "Semantic + ATS Analysis"),
    ("📄", "Bulk Resume Upload", "Supports up to 20 resumes"),
    ("🔍", "Retrieval-Augmented Generation", "Grounded AI Responses"),
    ("⚡", "Groq AI", "Ultra Fast Career Advisor"),
]
for col, (icon, title, desc) in zip([col1, col2, col3, col4], cards):
    with col:
        st.markdown(
            f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )
st.write("")

# ==========================================================
# Application Tabs
# ==========================================================

tab_screener, tab_advisor = st.tabs(["📑 Resume Screener", "🎯 AI Career Advisor"])

# ==========================================================
# Resume Screener Tab - FIXED with Full Logic
# ==========================================================

with tab_screener:
    st.subheader("📑 AI Resume Screening")
    st.info(
        """
**How it works**
1️⃣ Enter the Job Description
2️⃣ Upload one or more resumes
3️⃣ ResumeIQ performs Semantic Search + ATS Analysis
4️⃣ AI returns: Fit Score/100, Matched Skills, Missing Skills, 3 Suggestions
"""
    )

    left, right = st.columns(2)

    # ======================================================
    # Job Description - Direct Input Support
    # ======================================================
    with left:
        st.markdown("### 📄 Job Description Input")
        st.caption("Choose one method to provide the Job Description.")
        jd_method = st.radio("Choose Input Method", ["✍ Paste Job Description", "📁 Upload JD File"])
        jd_text = ""
        if jd_method == "✍ Paste Job Description":
            jd_text = st.text_area(
                "Paste Job Description",
                height=320,
                placeholder="Example:\nPython Developer\nRequired Skills\n• Python\n• FastAPI\n• AWS\n• Docker\n• PostgreSQL",
            )
        else:
            jd_file = st.file_uploader("Upload Job Description", type=["pdf", "docx"])
            if jd_file:
                parsed_jd = parse_document(jd_file.read(), jd_file.name)
                if parsed_jd.is_usable:
                    jd_text = parsed_jd.raw_text
                    st.success(f"✅ Successfully loaded {parsed_jd.filename}")
                else:
                    st.error(parsed_jd.error_message)

    # ======================================================
    # Resume Upload
    # ======================================================
    with right:
        st.markdown("### 📂 Resume Upload")
        st.caption("Upload up to 20 candidate resumes.")
        resume_files = st.file_uploader("Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True)
        if resume_files:
            if len(resume_files) > 20:
                st.warning("Only the first 20 resumes will be processed.")
                resume_files = resume_files[:20]
            st.success(f"✅ {len(resume_files)} resume(s) ready.")

    st.divider()
    run_clicked = st.button("🚀 Analyze Resumes", disabled=not (jd_text.strip() and resume_files))

    # ======================================================
    # ANALYSIS LOGIC - This was missing in your code
    # ======================================================
    if run_clicked:
        with st.spinner("🧠 Analyzing resumes with AI..."):
            jd_clean = clean_text(jd_text)
            results = []
            resume_texts = {}

            progress_bar = st.progress(0)
            for i, resume_file in enumerate(resume_files):
                parsed_resume = parse_document(resume_file.read(), resume_file.name)
                if not parsed_resume.is_usable:
                    st.error(f"Skipping {resume_file.name}: {parsed_resume.error_message}")
                    continue

                resume_clean = clean_text(parsed_resume.raw_text)
                resume_texts[parsed_resume.filename] = resume_clean

                # 1. CALCULATE SCORE - Returns 1 clear score out of 100
                match_result = score_resume(jd_clean, resume_clean, parsed_resume.filename)

                results.append({
                    "Filename": match_result.filename,
                    "Match %": match_result.final_score,
                    "Semantic %": match_result.semantic_score,
                    "Keyword %": match_result.keyword_score,
                    "Matched Skills": ", ".join(match_result.matched_skills) if match_result.matched_skills else "(none)",
                    "Missing Skills": ", ".join(match_result.missing_skills) if match_result.missing_skills else "(none)",
                })
                progress_bar.progress((i + 1) / len(resume_files))

            if results:
                df = pd.DataFrame(results).sort_values("Match %", ascending=False)
                st.session_state["results_df"] = df
                st.session_state["resume_texts"] = resume_texts
                st.session_state["jd_clean"] = jd_clean
                st.success("✅ Analysis Complete! Go to 'AI Career Advisor' tab for detailed insights.")
            else:
                st.error("No valid resumes processed.")

    # Show results table
    if "results_df" in st.session_state:
        st.markdown("### 📊 Screening Results")
        st.dataframe(st.session_state["results_df"], width="stretch")

        # Visualization
        df_viz = st.session_state["results_df"]
        fig = px.bar(df_viz, x="Filename", y="Match %", color="Match %", color_continuous_scale="Viridis")
        st.plotly_chart(fig, width="stretch")

# ==========================================================
# AI Career Advisor Tab - with RAG + Chat
# ==========================================================

with tab_advisor:
    st.subheader("🎯 AI Career Advisor with RAG")
    st.write("Generate personalized career advice for each candidate using Groq AI + Retrieval Augmented Generation.")

    if "results_df" not in st.session_state:
        st.info("👈 Please screen resumes first from the 'Resume Screener' tab.")
    else:
        df = st.session_state["results_df"]
        resume_texts = st.session_state["resume_texts"]
        jd_clean = st.session_state["jd_clean"]

        for _, row in df.iterrows():
            filename = row["Filename"]
            advice_key = f"advice_{filename}"

            with st.expander(f"🎯 {filename} • **{row['Match %']:.2f}% Match**"):
                col_btn, col_chat = st.columns([1, 1])

                with col_btn:
                    if st.button("🤖 Generate AI Advice", key=f"btn_{filename}"):
                        matched_skills = [] if row["Matched Skills"] == "(none)" else [s.strip() for s in row["Matched Skills"].split(",")]
                        missing_skills = [] if row["Missing Skills"] == "(none)" else [s.strip() for s in row["Missing Skills"].split(",")]

                        # Build MatchResult object for RAG
                        match_result = MatchResult(
                            filename=filename,
                            semantic_score=row["Semantic %"],
                            keyword_score=row["Keyword %"],
                            final_score=row["Match %"],
                            matched_skills=matched_skills,
                            missing_skills=missing_skills,
                        )

                        # Build RAG Context
                        context = build_context_block(match_result, jd_clean, resume_texts[filename])

                        with st.spinner("🤖 Generating AI Advice with Groq AI..."):
                            try:
                                advice = generate_advice(context) # Returns structured JSON with exactly 3 suggestions
                                groundedness = check_groundedness(context, advice)
                                st.session_state[advice_key] = {
                                    "advice": advice,
                                    "groundedness": groundedness,
                                    "context": context
                                }
                            except AdviceGenerationError as exc:
                                st.error(f"AI Generation Error: {exc}")
                            except Exception as exc:
                                st.error(f"Unexpected Error: {exc}")

                with col_chat:
                    # CHAT-BASED FOLLOW-UP QUESTIONS - Sir requirement
                    follow_q = st.text_input("💬 Ask follow-up question", key=f"chat_{filename}", placeholder="Ex: How to improve AWS skill?")
                    if follow_q and advice_key in st.session_state:
                        context = st.session_state[advice_key]["context"]
                        ans = ask_followup(context, follow_q)
                        st.info(f"**AI:** {ans}")

                # Display Advice if generated
                if advice_key in st.session_state:
                    cached = st.session_state[advice_key]
                    advice = cached["advice"]
                    groundedness = cached["groundedness"]

                    if groundedness.is_grounded:
                        st.success("✅ Groundedness Check Passed - AI skill references verified against ATS evidence")
                    else:
                        st.warning("⚠ Unverified skills detected: " + ", ".join(groundedness.unverified_skills))

                    st.markdown("#### 📌 Gap Analysis")
                    st.write(advice.gap_analysis.summary)
                    if advice.gap_analysis.missing_skill_gaps:
                        for item in advice.gap_analysis.missing_skill_gaps:
                            st.markdown(f"- **{item.skill}**")
                            st.markdown(f" - *Why it matters:* {item.why_it_matters}")
                            st.markdown(f" - *How to address:* {item.how_to_address}")
                    else:
                        st.info("No major skill gaps found!")

                    st.markdown("#### 💼 Interview Questions")
                    for q in advice.interview_questions:
                        st.markdown(f"- **Q:** {q.question}")
                        st.caption(f" *Tests:* {q.related_skill} | *Purpose:* {q.purpose}")

                    st.markdown("#### 📝 Top 3 Resume Suggestions")
                    for i, s in enumerate(advice.resume_suggestions, 1):
                        st.markdown(f"**{i}. {s.area}**")
                        st.markdown(f" *Suggestion:* {s.suggestion}")
                        st.markdown(f" *Rationale:* {s.rationale}")

st.markdown("---")
st.caption("📄 ResumeIQ Pro - AI Resume Screening System using RAG | Developed by Sangeetha Chirla | Powered by Groq AI")
