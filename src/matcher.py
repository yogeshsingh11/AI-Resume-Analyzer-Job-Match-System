"""
matcher.py
----------
Converts resume/job-description text into semantic embeddings and computes
a match score via cosine similarity, plus a rule-based skill gap analysis.

Pipeline position:
    Resume Text / Job Description -> Sentence Transformer -> Embeddings
        -> Cosine Similarity -> Match Score
                              -> Skill Gap Analysis
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and cache it for reuse."""
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> np.ndarray:
    """Turn a piece of text into a semantic embedding vector."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe against zero vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Compute an overall semantic match score (0-100) between a resume
    and a job description using sentence embeddings + cosine similarity.
    """
    resume_vec = embed_text(resume_text)
    jd_vec = embed_text(jd_text)
    similarity = cosine_similarity(resume_vec, jd_vec)
    # Cosine similarity for sentence embeddings tends to sit in a
    # compressed band (roughly 0.2-0.9), so rescale for a more
    # intuitive 0-100 "match percentage" without ever exceeding 100.
    score = max(0.0, min(1.0, similarity)) * 100
    return round(score, 1)


def skill_gap_analysis(resume_skills: List[str], jd_skills: List[str]) -> Dict:
    """Compare extracted resume skills against job-description skills."""
    resume_set = {s.lower() for s in resume_skills}
    jd_set = {s.lower() for s in jd_skills}

    matched = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)
    extra = sorted(resume_set - jd_set)

    coverage = round(100 * len(matched) / len(jd_set), 1) if jd_set else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "skill_coverage_pct": coverage,
    }


def experience_education_summary(resume_fields: Dict, jd_text: str) -> Dict:
    """
    Lightweight comparison of the resume's stated experience/education
    against whatever the job description appears to require.
    """
    jd_lower = jd_text.lower()
    years = resume_fields.get("years_experience")

    # crude requirement check: look for "X years" mentioned in the JD
    import re

    jd_year_matches = re.findall(r"(\d+)\+?\s*(?:years|yrs)", jd_lower)
    required_years = max((int(y) for y in jd_year_matches), default=None)

    meets_experience = None
    if years is not None and required_years is not None:
        meets_experience = years >= required_years

    degree_required = any(
        kw in jd_lower
        for kw in ["bachelor", "master", "b.tech", "m.tech", "degree", "mba", "phd"]
    )

    return {
        "candidate_years_experience": years,
        "required_years_experience": required_years,
        "meets_experience_requirement": meets_experience,
        "candidate_education": resume_fields.get("education", []),
        "degree_appears_required": degree_required,
    }
