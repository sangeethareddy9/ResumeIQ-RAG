"""

ResumeIQ - AI Resume Screening System using RAG


Author:
    Sangeetha Chirla

Description:
    ResumeIQ is an AI-powered Resume Screening System that
    compares resumes against a Job Description using
    semantic embeddings, keyword matching, and vector search.

Features:
    ✔ Resume Ranking
    ✔ Semantic Matching
    ✔ Keyword Matching
    ✔ Dashboard Analytics
    ✔ Candidate Skill Analysis
    ✔ AI Career Advisor (RAG + Ollama)
    ✔ CSV Export

Tech Stack:
    - Python
    - Streamlit
    - Sentence Transformers
    - spaCy
    - Plotly
    - Pandas
    - FAISS / Vector Search
    - Ollama

Run:
    streamlit run app/streamlit_app.py

"""

# ==========================================================
# Import Libraries
# ==========================================================

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow importing from src folder
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

# ==========================================================
# Import Project Modules
# ==========================================================

from src.parser import parse_document
from src.preprocess import clean_text
from src.embeddings import (
    embed_text,
    embed_batch,
)
from src.vector_index import ResumeVectorIndex
from src.skills_data import extract_skills_from_text
from src.scoring import MatchResult
from src.ats_score import calculate_ats_score

from src.rag_assistant import (
    build_context_block,
    generate_advice,
    check_groundedness,
    OllamaConnectionError,
    AdviceGenerationError,
)

# ==========================================================
# Streamlit Page Configuration
# ==========================================================

st.set_page_config(
    page_title="ResumeIQ",
    page_icon="📄",
    layout="wide",
)

st.title("📄 ResumeIQ")

st.caption(
    "AI Resume Screening System using Retrieval-Augmented Generation (RAG)"
)

st.markdown("---")

# ==========================================================
# Application Tabs
# ==========================================================

tab_screener, tab_advisor = st.tabs(
    [
        "📑 Resume Screener",
        "🎯 AI Career Advisor",
    ]
)

# ==========================================================
# Resume Screener
# ==========================================================

with tab_screener:

    st.caption(
        """
        Upload a Job Description and multiple resumes.
        ResumeIQ compares resumes using semantic embeddings,
        keyword matching, and vector similarity to rank
        candidates automatically.
        """
    )

    jd_column, resume_column = st.columns(2)

    # ======================================================
    # Job Description Section
    # ======================================================

    with jd_column:

        st.subheader("📄 Job Description")

        jd_input_method = st.radio(
            "Choose Job Description Input",
            ["Paste Text", "Upload File"],
            horizontal=True
        )

        jd_text = ""

        if jd_input_method == "Paste Text":

            jd_text = st.text_area(
                "Paste the complete Job Description",
                height=260,
                placeholder="""
Example:

We are looking for a Python Full Stack Developer.

Required Skills:
- Python
- Flask
- MySQL
- REST API
- HTML
- CSS
- JavaScript
- React
- Git
- SQL
"""
            )

        else:

            jd_file = st.file_uploader(
                "Upload Job Description",
                type=["pdf", "docx"]
            )

            if jd_file is not None:

                parsed_jd = parse_document(jd_file.read(), jd_file.name)

                if parsed_jd.is_usable:
                    jd_text = parsed_jd.raw_text
                    st.success(f"✅ Loaded {parsed_jd.char_count} characters")
                else:
                    st.error(parsed_jd.error_message)

    # ======================================================
    # Resume Upload Section
    # ======================================================

    with resume_column:

        st.subheader("📂 Upload Resumes")

        resume_files = st.file_uploader(
            "Upload up to 20 resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )

        if resume_files:

            if len(resume_files) > 20:
                st.warning("Only the first 20 resumes will be processed.")
                resume_files = resume_files[:20]

            st.success(f"✅ {len(resume_files)} Resume(s) Ready for Screening")

    # ======================================================
    # Start Screening Button
    # ======================================================

    run_clicked = st.button(
        "🚀 Screen Resumes",
        type="primary",
        disabled=not (jd_text.strip() and resume_files)
    )

    # ======================================================
    # Resume Parsing & Screening Engine
    # ======================================================

    if run_clicked:

        with st.spinner("📄 Reading uploaded resumes..."):

            filenames = []
            cleaned_texts = []
            skipped_files = []

            for resume in resume_files:

                parsed_resume = parse_document(resume.getvalue(), resume.name)

                if parsed_resume.is_usable:
                    filenames.append(parsed_resume.filename)
                    cleaned_texts.append(clean_text(parsed_resume.raw_text))
                else:
                    skipped_files.append(f"{parsed_resume.filename} ({parsed_resume.status.value})")

        for file in skipped_files:
            st.warning(f"⚠️ Skipped: {file}")

        if not filenames:
            st.error("No readable resumes were found.")
            st.stop()

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

        # ==================================================
        # Calculate Scores
        # ==================================================

        for filename, semantic_score in ranked_candidates:

            resume_skills = set(extract_skills_from_text(resume_text_lookup[filename]))
            matched_skills = sorted(jd_skills & resume_skills)
            missing_skills = sorted(jd_skills - resume_skills)

            ats_result = calculate_ats_score(resume_text_lookup[filename], jd_skills, resume_skills)
            ats_score = ats_result["score"]
            ats_breakdown = ats_result["breakdown"]

            keyword_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else semantic_score
            final_score = round(semantic_score * 0.70 + keyword_score * 0.30, 2)

            results.append(
                {
                    "Filename": filename,
                    "Match %": final_score,
                    "ATS Score": ats_score,
                    "ATS Breakdown": ats_breakdown,
                    "Semantic %": round(semantic_score, 2),
                    "Keyword %": round(keyword_score, 2),
                    "Matched Skills": ", ".join(matched_skills) if matched_skills else "(none)",
                    "Missing Skills": ", ".join(missing_skills) if missing_skills else "(none)"
                }
            )

        results_df = (
            pd.DataFrame(results)
          .sort_values("Match %", ascending=False)
          .reset_index(drop=True)
        )
        results_df.index += 1

        st.session_state["results_df"] = results_df
        st.session_state["resume_texts"] = resume_text_lookup
        st.session_state["jd_clean"] = jd_clean

    # ======================================================
    # Display Screening Results
    # ======================================================

    if "results_df" in st.session_state:

        df = st.session_state["results_df"]

        INTERVIEW_THRESHOLD = 60
        total_candidates = len(df)
        best_match = df["Match %"].max()
        average_match = round(df["Match %"].mean(), 2)
        selected_candidates = len(df[df["Match %"] >= INTERVIEW_THRESHOLD])

        metric1, metric2, metric3, metric4 = st.columns(4)
        with metric1: st.metric("👥 Candidates", total_candidates)
        with metric2: st.metric("🏆 Best Match", f"{best_match:.2f}%")
        with metric3: st.metric("📊 Average Score", f"{average_match:.2f}%")
        with metric4: st.metric("✅ Selected", selected_candidates)

        st.markdown("---")

        best_candidate = df.iloc[0]
        st.success(f"""🏆 **Top Candidate**\n\n**{best_candidate['Filename']}**\n\nOverall Match Score : **{best_candidate['Match %']:.2f}%**""")

        st.subheader("📋 Ranked Results")
        display_df = df[["Filename", "Match %", "Semantic %", "ATS Score", "Keyword %"]]

        search_name = st.text_input("🔍 Search Candidate", placeholder="Enter candidate filename...")
        filtered_df = display_df.copy()
        if search_name.strip():
            filtered_df = filtered_df[filtered_df["Filename"].str.contains(search_name, case=False, na=False)]

        st.dataframe(filtered_df, width="stretch")

        st.subheader("📈 Candidate Ranking")
        ranking_chart = px.bar(
            df.sort_values("Match %", ascending=True),
            x="Match %", y="Filename", orientation="h",
            color="Match %", color_continuous_scale="Blues", text="Match %"
        )
        ranking_chart.update_layout(height=max(350, len(df) * 55), xaxis_title="Match Score", yaxis_title="")
        st.plotly_chart(ranking_chart, width="stretch", key="main_ranking_chart")

        st.subheader("📊 Recruitment Summary")
        high = len(df[df["Match %"] >= 80])
        medium = len(df[(df["Match %"] >= 60) & (df["Match %"] < 80)])
        low = len(df[df["Match %"] < 60])

        summary_df = pd.DataFrame(
            {"Recommendation": ["Highly Recommended", "Recommended", "Needs Improvement"], "Candidates": [high, medium, low]}
        )
        summary_chart = px.pie(summary_df, names="Recommendation", values="Candidates", hole=0.45, title="Candidate Distribution")
        st.plotly_chart(summary_chart, width="stretch", key="summary_pie_chart")

        # ======================================================
        # Candidate Detailed Analysis
        # ======================================================

        st.subheader("👤 Candidate Breakdown")

        for _, row in df.iterrows():

            with st.expander(f"{row['Filename']} • {row['Match %']:.2f}% Match"):

                if row["Match %"] >= 80:
                    st.success("🟢 Highly Recommended")
                elif row["Match %"] >= 60:
                    st.info("🟡 Recommended for Interview")
                else:
                    st.error("🔴 Needs Improvement")

                st.subheader("📈 Resume Strength")
                st.progress(float(row["Match %"]) / 100)
                st.metric("Overall Match Score", f"{row['Match %']:.2f}%")
                st.metric("ATS Resume Score", f"{row['ATS Score']}/100")
                st.progress(row["ATS Score"] / 100)

                st.subheader("📋 ATS Score Breakdown")
                breakdown = row["ATS Breakdown"]
                for section, value in breakdown.items():
                    st.write(f"**{section} :** {value}")

                if row["ATS Score"] >= 85:
                    st.success("🌟 Excellent ATS Resume")
                elif row["ATS Score"] >= 70:
                    st.info("👍 Good ATS Resume")
                else:
                    st.warning("⚠️ ATS Resume Needs Improvement")

                score_col1, score_col2 = st.columns(2)
                with score_col1: st.metric("Semantic Score", f"{row['Semantic %']:.2f}%")
                with score_col2: st.metric("Keyword Score", f"{row['Keyword %']:.2f}%")

                st.subheader("🧠 Skill Analysis")
                if row["Keyword %"] >= 70:
                    st.success("Strong keyword alignment with the Job Description.")
                else:
                    st.warning("Some required technical skills are missing.")

                if row["Semantic %"] >= 70:
                    st.success("Resume content closely matches the job role.")
                else:
                    st.info("Improve project descriptions and work experience.")

                st.markdown("### ✅ Matched Skills")
                st.write(row["Matched Skills"])
                st.markdown("### ❌ Missing Skills")
                st.write(row["Missing Skills"])

                # ---------------------------------------------
                # Skill Match Visualization - FIXED WITH UNIQUE KEY
                # ---------------------------------------------
                matched_count = 0 if row["Matched Skills"] == "(none)" else len(row["Matched Skills"].split(", "))
                missing_count = 0 if row["Missing Skills"] == "(none)" else len(row["Missing Skills"].split(", "))

                if matched_count + missing_count > 0:
                    skill_df = pd.DataFrame(
                        {"Category": ["Matched Skills", "Missing Skills"], "Count": [matched_count, missing_count]}
                    )
                    skill_chart = px.pie(skill_df, names="Category", values="Count", hole=0.45, title="Skill Match Analysis")

                    st.plotly_chart(
                        skill_chart,
                        width="stretch",
                        key=f"skill_chart_{row['Filename']}" # FIX: unique key for each candidate
                    )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Screening Results",
            data=csv,
            file_name="resume_screening_results.csv",
            mime="text/csv"
        )

# ==========================================================
# AI Career Advisor (RAG + Ollama)
# ==========================================================

with tab_advisor:

    st.header("🎯 AI Career Advisor")
    st.write("Generate AI-powered career advice for each candidate using Retrieval-Augmented Generation (RAG).")

    if "results_df" not in st.session_state:
        st.info("Please screen resumes first from the Resume Screener tab.")
    else:
        df = st.session_state["results_df"]
        resume_texts = st.session_state["resume_texts"]
        jd_clean = st.session_state["jd_clean"]

        for _, row in df.iterrows():

            filename = row["Filename"]
            advice_key = f"advice_{filename}"

            with st.expander(f"🎯 {filename} ({row['Match %']:.2f}% Match)"):

                if st.button("🤖 Generate AI Advice", key=f"btn_{filename}"):

                    matched_skills = [] if row["Matched Skills"] == "(none)" else row["Matched Skills"].split(", ")
                    missing_skills = [] if row["Missing Skills"] == "(none)" else row["Missing Skills"].split(", ")

                    match_result = MatchResult(
                        filename=filename,
                        semantic_score=row["Semantic %"],
                        keyword_score=row["Keyword %"],
                        final_score=row["Match %"],
                        matched_skills=matched_skills,
                        missing_skills=missing_skills
                    )

                    context = build_context_block(match_result, jd_clean, resume_texts[filename])

                    with st.spinner("🤖 Generating AI Advice..."):
                        try:
                            advice = generate_advice(context)
                            groundedness = check_groundedness(context, advice)
                            st.session_state[advice_key] = {"advice": advice, "groundedness": groundedness}
                        except OllamaConnectionError as exc:
                            st.error(f"Ollama Error: {exc}")
                        except AdviceGenerationError as exc:
                            st.error(f"Advice Error: {exc}")

                if advice_key in st.session_state:
                    cached = st.session_state[advice_key]
                    advice = cached["advice"]
                    groundedness = cached["groundedness"]

                    if groundedness.is_grounded:
                        st.success("✅ Groundedness Check Passed")
                    else:
                        st.warning("⚠ Some generated skills could not be verified.\n\n" + ", ".join(groundedness.unverified_skills))

                    st.subheader("📌 Gap Analysis")
                    gap = advice["gap_analysis"]
                    st.write(gap["summary"])

                    if gap["missing_skill_gaps"]:
                        for item in gap["missing_skill_gaps"]:
                            st.markdown(f"### {item['skill']}\n\n**Why it Matters**\n{item['why_it_matters']}\n\n**How to Improve**\n{item['how_to_address']}")
                    else:
                        st.success("No major skill gaps found.")

                    st.subheader("💼 Suggested Interview Questions")
                    for question in advice["interview_questions"]:
                        st.markdown(f"### Question\n{question['question']}\n\n**Skill Tested**\n{question['related_skill']}\n\n**Purpose**\n{question['purpose']}")

                    st.subheader("📝 Resume Improvement Suggestions")
                    for suggestion in advice["resume_suggestions"]:
                        st.markdown(f"### {suggestion['area']}\n\n**Suggestion**\n{suggestion['suggestion']}\n\n**Reason**\n{suggestion['rationale']}")

# ==========================================================
# End of ResumeIQ Application
# ==========================================================

st.markdown("---")
st.caption("📄 ResumeIQ - AI Resume Screening System using RAG")
st.caption("Developed by Sangeetha Chirla")