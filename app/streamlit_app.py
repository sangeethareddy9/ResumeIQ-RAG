"""
ResumeIQ Pro - AI Resume Screening System using RAG + Groq

Author: Sangeetha Chirla
Tech: Python, Streamlit, Sentence Transformers, FAISS, Groq Llama 3.3
Run: streamlit run app/streamlit_app.py
"""

# ==========================================================
# Import Libraries
# ==========================================================

import sys
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
from src.embeddings import embed_text, embed_batch
from src.vector_index import ResumeVectorIndex
from src.skills_data import extract_skills_from_text
from src.scoring import MatchResult
from src.ats_score import calculate_ats_score
from src.rag_assistant import (
    build_context_block,
    generate_advice,
    check_groundedness,
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

    /* Hide default elements */
    #MainMenu, footer, header {visibility: hidden;}

    /* BACKGROUND - BLACK */
  .stApp {
        background: #0a0a0a;
        background-image: radial-gradient(at 20% 0%, #4c1d95 0px, transparent 50%),
                          radial-gradient(at 80% 0%, #7c3aed 0px, transparent 50%);
    }

    /* MAIN CONTAINER - DARK CARD */
  .block-container {
        background: #111;
        border: 1px solid #262626;
        padding: 2.5rem 3rem;
        border-radius: 20px;
        box-shadow: 0 0 40px rgba(124, 58, 237, 0.15);
    }

    /* HERO - VIOLET GRADIENT */
  .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a855f7, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
  .hero-subtitle {
        text-align: center;
        color: #a1a1aa!important;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
    }

    /* FEATURE CARDS - DARK */
  .feature-card {
        background: #18181b;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #27272a;
        text-align: center;
        transition: all 0.3s ease;
    }
  .feature-card:hover {
        transform: translateY(-5px);
        border-color: #7c3aed;
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.3);
    }
  .feature-icon {font-size: 2.2rem;}
  .feature-title {font-weight: 600; color: #fafafa!important; margin-top: 0.5rem;}
  .feature-desc {font-size: 0.85rem; color: #a1a1aa!important;}

    /* BUTTON - VIOLET */
  .stButton>button {
        background: linear-gradient(90deg, #7c3aed, #a855f7)!important;
        color: white!important; border-radius: 10px; border: none;
        padding: 0.8rem 1.5rem; font-weight: 700; width: 100%;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.4);
    }
  .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(168, 85, 247, 0.6);
    }

    /* FIX UPLOADER - DARK */
    [data-testid="stFileUploader"] {
        background: #18181b!important; 
        border: 2px dashed #3f3f46!important;
        border-radius: 12px!important;
    }
    [data-testid="stFileUploader"] button {
        background: #7c3aed!important; color: white!important;
    }
    [data-testid="stFileUploader"] * {color: #d4d4d8!important;}

    /* FIX TEXTAREA - DARK */
    textarea {
        background: #18181b!important; color: #fafafa!important;
        border: 2px solid #3f3f46!important; border-radius: 12px!important;
    }
    textarea:focus {
        border-color: #7c3aed!important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2)!important;
    }

    /* FIX RADIO BUTTONS */
    [data-testid="stRadio"] label {
        color: #d4d4d8!important; font-weight: 500!important;
    }

    /* TABS - DARK WITH GLOW */
  .stTabs [data-baseweb="tab-list"] {
        background: #18181b; border-radius: 10px; padding: 4px;
        border: 1px solid #27272a;
    }
  .stTabs [data-baseweb="tab"] {
        color: #a1a1aa!important; font-weight: 600; border-radius: 8px;
    }
  .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7c3aed, #a855f7);
        color: white!important;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.5);
    }

    /* METRICS - DARK */
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
st.markdown('<p class="hero-subtitle">AI-Powered Resume Screening & Career Advisor using RAG + Groq Llama 3.3</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
features = [
    ("🧠", "AI Powered", "Semantic + Keyword Matching"),
    ("📄", "20 Resumes", "Bulk Processing Limit"),
    ("🔍", "RAG Search", "Context-Aware Analysis"),
    ("⚡", "Groq Llama 3.3", "Ultra-Fast AI Advisor")
]
for i, (icon, title, desc) in enumerate(features):
    with [col1, col2, col3, col4][i]:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ==========================================================
# Application Tabs
# ==========================================================

tab_screener, tab_advisor = st.tabs(["📑 Resume Screener", "🎯 AI Career Advisor"])

# ==========================================================
# Resume Screener Tab
# ==========================================================

with tab_screener:
    st.subheader("Start Screening Candidates")

    jd_column, resume_column = st.columns(2, gap="large")

    with jd_column:
        st.markdown("#### 📄 Job Description")
        jd_input_method = st.radio("Input Method", ["Paste Text", "Upload File"], horizontal=True, label_visibility="collapsed")
        jd_text = ""
        if jd_input_method == "Paste Text":
            jd_text = st.text_area("JD Input", height=280, placeholder="Paste complete Job Description here...\n\nRequired Skills:\n- Python\n- Flask\n- SQL\n- React", label_visibility="collapsed")
        else:
            jd_file = st.file_uploader("Upload JD", type=["pdf", "docx"], label_visibility="collapsed")
            if jd_file:
                parsed_jd = parse_document(jd_file.read(), jd_file.name)
                if parsed_jd.is_usable:
                    jd_text = parsed_jd.raw_text
                    st.success(f"✅ Loaded {parsed_jd.char_count} characters")
                else:
                    st.error(parsed_jd.error_message)

    with resume_column:
        st.markdown("#### 📂 Upload Resumes")
        resume_files = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed")
        if resume_files:
            if len(resume_files) > 20:
                st.warning("Only first 20 resumes will be processed.")
                resume_files = resume_files[:20]
            st.info(f"✅ {len(resume_files)} Resume(s) Ready")

    run_clicked = st.button("🚀 Screen Resumes", disabled=not (jd_text.strip() and resume_files))

    if run_clicked:
        with st.spinner("📄 Reading uploaded resumes..."):
            filenames, cleaned_texts, skipped_files = [], [], []
            for resume in resume_files:
                parsed_resume = parse_document(resume.getvalue(), resume.name)
                if parsed_resume.is_usable:
                    filenames.append(parsed_resume.filename)
                    cleaned_texts.append(clean_text(parsed_resume.raw_text))
                else:
                    skipped_files.append(f"{parsed_resume.filename}")

        for file in skipped_files: st.warning(f"⚠️ Skipped: {file}")
        if not filenames: st.error("No readable resumes were found."); st.stop()

        with st.spinner("🧠 Analyzing resumes using AI..."):
            jd_clean = clean_text(jd_text)
            jd_embedding = embed_text(jd_clean)
            jd_skills = set(extract_skills_from_text(jd_clean))
            resume_embeddings = embed_batch(cleaned_texts)
            vector_index = ResumeVectorIndex(embedding_dim=resume_embeddings.shape[1])
            vector_index.build(filenames, resume_embeddings)
            ranked_candidates = vector_index.search(jd_embedding)
            resume_text_lookup = dict(zip(filenames, cleaned_texts))
            results = []

        for filename, semantic_score in ranked_candidates:
            resume_skills = set(extract_skills_from_text(resume_text_lookup[filename]))
            matched_skills = sorted(jd_skills & resume_skills)
            missing_skills = sorted(jd_skills - resume_skills)
            ats_result = calculate_ats_score(resume_text_lookup[filename], jd_skills, resume_skills)
            keyword_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else semantic_score
            final_score = round(semantic_score * 0.70 + keyword_score * 0.30, 2)
            results.append({
                "Filename": filename, "Match %": final_score, "ATS Score": ats_result["score"],
                "ATS Breakdown": ats_result["breakdown"], "Semantic %": round(semantic_score, 2),
                "Keyword %": round(keyword_score, 2), "Matched Skills": ", ".join(matched_skills) if matched_skills else "(none)",
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "(none)"
            })

        results_df = pd.DataFrame(results).sort_values("Match %", ascending=False).reset_index(drop=True)
        results_df.index += 1
        st.session_state["results_df"] = results_df
        st.session_state["resume_texts"] = resume_text_lookup
        st.session_state["jd_clean"] = jd_clean

    if "results_df" in st.session_state:
        df = st.session_state["results_df"]
        st.divider()
        st.subheader("📊 Dashboard Analytics")

        INTERVIEW_THRESHOLD = 60
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("👥 Candidates", len(df))
        m2.metric("🏆 Best Match", f"{df['Match %'].max():.2f}%")
        m3.metric("📊 Average Score", f"{df['Match %'].mean():.2f}%")
        m4.metric("✅ Selected", len(df[df["Match %"] >= INTERVIEW_THRESHOLD]))

        st.success(f"🏆 **Top Candidate: {df.iloc[0]['Filename']}** with **{df.iloc[0]['Match %']:.2f}%** Match")

        st.subheader("📋 Ranked Results")
        search_name = st.text_input("🔍 Search Candidate", placeholder="Enter candidate filename...")
        display_df = df[["Filename", "Match %", "Semantic %", "ATS Score", "Keyword %"]]
        if search_name.strip(): display_df = display_df[display_df["Filename"].str.contains(search_name, case=False, na=False)]
        st.dataframe(display_df, width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📈 Candidate Ranking")
            ranking_chart = px.bar(df.sort_values("Match %", ascending=True), x="Match %", y="Filename", orientation="h", color="Match %", color_continuous_scale="Blues", text="Match %")
            ranking_chart.update_layout(height=max(350, len(df) * 50), margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(ranking_chart, width="stretch", key="main_ranking_chart")

        with c2:
            st.subheader("📊 Distribution")
            high = len(df[df["Match %"] >= 80]); medium = len(df[(df["Match %"] >= 60) & (df["Match %"] < 80)]); low = len(df[df["Match %"] < 60])
            summary_df = pd.DataFrame({"Recommendation": ["Highly Recommended", "Recommended", "Needs Improvement"], "Candidates": [high, medium, low]})
            summary_chart = px.pie(summary_df, names="Recommendation", values="Candidates", hole=0.5, color_discrete_sequence=px.colors.sequential.Blues)
            summary_chart.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(summary_chart, width="stretch", key="summary_pie_chart")

        st.subheader("👤 Candidate Detailed Breakdown")
        for _, row in df.iterrows():
            with st.expander(f"{row['Filename']} • {row['Match %']:.2f}% Match"):
                if row["Match %"] >= 80: st.success("🟢 Highly Recommended")
                elif row["Match %"] >= 60: st.info("🟡 Recommended for Interview")
                else: st.error("🔴 Needs Improvement")

                colA, colB, colC = st.columns(3)
                colA.metric("Overall Match", f"{row['Match %']:.2f}%")
                colB.metric("ATS Score", f"{row['ATS Score']}/100")
                colC.metric("Semantic", f"{row['Semantic %']:.2f}%")

                st.markdown("**📋 ATS Breakdown**")
                for section, value in row["ATS Breakdown"].items(): st.progress(value/100, text=f"{section}: {value}")

                st.markdown("**🧠 Skills**")
                sc1, sc2 = st.columns(2)
                sc1.markdown(f"**✅ Matched:**\n{row['Matched Skills']}")
                sc2.markdown(f"**❌ Missing:**\n{row['Missing Skills']}")

                matched_count = 0 if row["Matched Skills"] == "(none)" else len(row["Matched Skills"].split(", "))
                missing_count = 0 if row["Missing Skills"] == "(none)" else len(row["Missing Skills"].split(", "))
                if matched_count + missing_count > 0:
                    skill_df = pd.DataFrame({"Category": ["Matched", "Missing"], "Count": [matched_count, missing_count]})
                    skill_chart = px.pie(skill_df, names="Category", values="Count", hole=0.6, color_discrete_sequence=["#4ade80", "#f87171"])
                    st.plotly_chart(skill_chart, width="stretch", key=f"skill_chart_{row['Filename']}")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Screening Results", data=csv, file_name="resume_screening_results.csv", mime="text/csv")

# ==========================================================
# AI Career Advisor Tab - FIXED MatchResult
# ==========================================================

with tab_advisor:
    st.subheader("🎯 AI Career Advisor")
    st.write("Generate personalized career advice for each candidate using Groq Llama 3.3 + RAG.")

    if "results_df" not in st.session_state:
        st.info("Please screen resumes first from the Resume Screener tab.")
    else:
        df = st.session_state["results_df"]
        resume_texts = st.session_state["resume_texts"]
        jd_clean = st.session_state["jd_clean"]

        for _, row in df.iterrows():
            filename = row["Filename"]
            advice_key = f"advice_{filename}"
            with st.expander(f"🎯 {filename} • {row['Match %']:.2f}% Match"):
                if st.button("🤖 Generate AI Advice", key=f"btn_{filename}"):
                    matched_skills = [] if row["Matched Skills"] == "(none)" else row["Matched Skills"].split(", ")
                    missing_skills = [] if row["Missing Skills"] == "(none)" else row["Missing Skills"].split(", ")
                    match_result = MatchResult(
                        filename=filename,
                        semantic_score=row["Semantic %"],
                        keyword_score=row["Keyword %"],
                        final_score=row["Match %"],
                        matched_skills=matched_skills,
                        missing_skills=missing_skills,
                    )
                    context = build_context_block(match_result, jd_clean, resume_texts[filename])
                    with st.spinner("🤖 Generating AI Advice with Groq..."):
                        try:
                            advice = generate_advice(context)
                            groundedness = check_groundedness(context, advice)
                            st.session_state[advice_key] = {"advice": advice, "groundedness": groundedness}
                        except AdviceGenerationError as exc: st.error(f"AI Error: {exc}")
                        except Exception as exc: st.error(f"Unexpected Error: {exc}")

                if advice_key in st.session_state:
                    cached = st.session_state[advice_key]; advice = cached["advice"]; groundedness = cached["groundedness"]
                    if groundedness.is_grounded: st.success("✅ Groundedness Check Passed")
                    else: st.warning("⚠ Unverified skills: " + ", ".join(groundedness.unverified_skills))

                    st.markdown("#### 📌 Gap Analysis")
                    st.write(advice["gap_analysis"]["summary"])
                    for item in advice["gap_analysis"]["missing_skill_gaps"]:
                        st.markdown(f"**{item['skill']}**\n- *Why:* {item['why_it_matters']}\n- *How:* {item['how_to_address']}")

                    st.markdown("#### 💼 Interview Questions")
                    for q in advice["interview_questions"]: st.markdown(f"- **Q:** {q['question']} \n *Skill:* {q['related_skill']}")

                    st.markdown("#### 📝 Resume Suggestions")
                    for s in advice["resume_suggestions"]: st.markdown(f"- **{s['area']}:** {s['suggestion']}")

st.markdown("---")
st.caption("📄 ResumeIQ Pro - AI Resume Screening System using RAG | Developed by Sangeetha Chirla")