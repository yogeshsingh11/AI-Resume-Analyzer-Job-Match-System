"""
AI Resume Analyzer & Job Match System
--------------------------------------
Streamlit front-end that ties together the full pipeline:

    PDF -> Text Extraction -> NLP -> Embeddings -> Semantic Similarity
        -> Match Score -> Skill Gap Analysis -> LLM -> Recommendations
        -> Streamlit Dashboard
"""

from __future__ import annotations

import streamlit as st

from src.pdf_parser import extract_text_from_pdf, PDFParsingError
from src.nlp_processor import process_document
from src.matcher import compute_match_score, skill_gap_analysis, experience_education_summary
from src.llm_analyzer import generate_analysis, fallback_analysis, LLMAnalysisError, DEFAULT_MODEL

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        help="Needed for AI-generated strengths/weaknesses, recommendations, "
        "and interview questions. Without it, the app still shows the match "
        "score and skill-gap analysis using a basic fallback for the rest.",
    )
    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.divider()
    st.caption(
        "Pipeline: PDF → Text Extraction → NLP → Embeddings → "
        "Semantic Similarity → Match Score → Skill Gap Analysis → "
        "LLM → Recommendations"
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 AI Resume Analyzer & Job Match System")
st.write(
    "Upload a resume and paste a job description to get a match score, "
    "skill-gap breakdown, and AI-generated feedback."
)

col_left, col_right = st.columns(2)

with col_left:
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col_right:
    jd_text_input = st.text_area("Paste Job Description", height=220)

analyze_clicked = st.button("🔍 ANALYZE RESUME", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyze_clicked:
    if resume_file is None:
        st.error("Please upload a resume PDF first.")
        st.stop()
    if not jd_text_input.strip():
        st.error("Please paste a job description first.")
        st.stop()

    with st.spinner("Extracting text from resume..."):
        try:
            resume_text = extract_text_from_pdf(resume_file)
        except PDFParsingError as exc:
            st.error(str(exc))
            st.stop()

    with st.spinner("Running NLP extraction..."):
        resume_fields = process_document(resume_text)
        jd_fields = process_document(jd_text_input)

    with st.spinner("Computing semantic match score..."):
        match_score = compute_match_score(resume_text, jd_text_input)

    gap = skill_gap_analysis(resume_fields["skills"], jd_fields["skills"])
    exp_edu = experience_education_summary(resume_fields, jd_text_input)

    with st.spinner("Generating AI recommendations..."):
        if api_key:
            try:
                llm_result = generate_analysis(
                    resume_text=resume_text,
                    jd_text=jd_text_input,
                    match_score=match_score,
                    matched_skills=gap["matched_skills"],
                    missing_skills=gap["missing_skills"],
                    api_key=api_key,
                    model=model or DEFAULT_MODEL,
                )
            except LLMAnalysisError as exc:
                st.warning(f"LLM analysis unavailable, showing basic fallback instead. ({exc})")
                llm_result = fallback_analysis(gap["matched_skills"], gap["missing_skills"])
        else:
            llm_result = fallback_analysis(gap["matched_skills"], gap["missing_skills"])

    st.divider()

    # --- Match score ---------------------------------------------------
    st.subheader("Match Score")
    score_col, bar_col = st.columns([1, 3])
    with score_col:
        st.metric("Overall Match", f"{match_score}%")
    with bar_col:
        st.progress(min(int(match_score), 100) / 100)

    # --- Skills ----------------------------------------------------------
    st.subheader("Skills")
    skill_col1, skill_col2 = st.columns(2)
    with skill_col1:
        st.markdown("**✅ Matched Skills**")
        if gap["matched_skills"]:
            st.markdown(", ".join(f"`{s}`" for s in gap["matched_skills"]))
        else:
            st.caption("No overlapping skills detected.")
    with skill_col2:
        st.markdown("**❌ Missing Skills**")
        if gap["missing_skills"]:
            st.markdown(", ".join(f"`{s}`" for s in gap["missing_skills"]))
        else:
            st.caption("No obvious gaps detected.")
    st.caption(f"Skill coverage: {gap['skill_coverage_pct']}% of skills mentioned in the JD")

    # --- Try a nice chart if plotly is available --------------------------
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Bar(
                x=["Matched", "Missing"],
                y=[len(gap["matched_skills"]), len(gap["missing_skills"])],
                marker_color=["#2ecc71", "#e74c3c"],
            )
        )
        fig.update_layout(title="Skill Match Breakdown", height=320)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass

    # --- Experience / education ------------------------------------------
    st.subheader("Experience & Education")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        yrs = exp_edu["candidate_years_experience"]
        req = exp_edu["required_years_experience"]
        st.write(f"**Candidate experience:** {yrs if yrs is not None else 'Not stated'} years")
        st.write(f"**JD appears to require:** {req if req is not None else 'Not specified'} years")
        if exp_edu["meets_experience_requirement"] is not None:
            verdict = "✅ Meets requirement" if exp_edu["meets_experience_requirement"] else "⚠️ Below requirement"
            st.write(f"**Verdict:** {verdict}")
    with exp_col2:
        edu = exp_edu["candidate_education"]
        st.write(f"**Candidate education keywords:** {', '.join(edu) if edu else 'Not detected'}")
        st.write(f"**Degree appears required by JD:** {'Yes' if exp_edu['degree_appears_required'] else 'No / unclear'}")

    # --- Strengths / weaknesses --------------------------------------------
    st.subheader("Resume Strengths & Weaknesses")
    sw_col1, sw_col2 = st.columns(2)
    with sw_col1:
        st.markdown("**💪 Strengths**")
        for item in llm_result.get("strengths", []):
            st.markdown(f"- {item}")
    with sw_col2:
        st.markdown("**⚠️ Weaknesses**")
        for item in llm_result.get("weaknesses", []):
            st.markdown(f"- {item}")

    # --- Recommendations ------------------------------------------------
    st.subheader("🤖 AI-Generated Improvement Suggestions")
    for item in llm_result.get("recommendations", []):
        st.markdown(f"- {item}")

    # --- Interview questions ---------------------------------------------
    st.subheader("🎤 Relevant Interview Questions")
    for i, question in enumerate(llm_result.get("interview_questions", []), start=1):
        st.markdown(f"{i}. {question}")

    # --- Contact info sanity check -----------------------------------------
    with st.expander("Extracted resume contact details"):
        st.write(f"Email: {resume_fields['email'] or 'Not found'}")
        st.write(f"Phone: {resume_fields['phone'] or 'Not found'}")
        st.write(f"Organizations mentioned: {', '.join(resume_fields['organizations']) or 'None detected'}")
