"""
llm_analyzer.py
---------------
Calls an LLM (Anthropic's Claude by default) to produce the qualitative
parts of the analysis: strengths, weaknesses, improvement recommendations,
and tailored interview questions.

Pipeline position:
    Match Score + Skill Gap Analysis -> LLM Analysis -> Recommendations
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an expert technical recruiter and resume coach.
You will be given a candidate's resume text, a job description, a computed
semantic match score, and lists of matched/missing skills.

Analyze the fit and respond with ONLY a single valid JSON object — no
markdown fences, no commentary before or after — with exactly this shape:

{
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "recommendations": ["...", "..."],
  "interview_questions": ["...", "..."]
}

Guidelines:
- "strengths": 3-5 concrete things the resume does well relative to this job.
- "weaknesses": 3-5 concrete gaps or unclear areas relative to this job.
- "recommendations": 4-6 specific, actionable edits to improve the resume
  for THIS job (not generic resume advice).
- "interview_questions": 5-7 questions an interviewer would plausibly ask
  this candidate for this role, mixing technical and behavioral questions.
- Be specific and reference actual details from the resume/job description
  where possible. Do not invent facts not supported by the text.
"""


class LLMAnalysisError(Exception):
    """Raised when the LLM call fails or returns unparsable output."""


def _build_user_prompt(
    resume_text: str,
    jd_text: str,
    match_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
) -> str:
    return f"""Match score: {match_score}%
Matched skills: {", ".join(matched_skills) if matched_skills else "none detected"}
Missing skills: {", ".join(missing_skills) if missing_skills else "none detected"}

RESUME:
\"\"\"
{resume_text[:8000]}
\"\"\"

JOB DESCRIPTION:
\"\"\"
{jd_text[:4000]}
\"\"\"

Return the JSON object now."""


def _extract_json(raw_text: str) -> Dict:
    """Best-effort extraction of a JSON object from the model's raw reply."""
    text = raw_text.strip()
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to grabbing the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def generate_analysis(
    resume_text: str,
    jd_text: str,
    match_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> Dict:
    """
    Call the LLM to generate strengths, weaknesses, recommendations, and
    interview questions. Returns a dict with those four keys.
    """
    if not api_key:
        raise LLMAnalysisError("No API key provided.")

    client = anthropic.Anthropic(api_key=api_key)

    user_prompt = _build_user_prompt(
        resume_text, jd_text, match_score, matched_skills, missing_skills
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.APIError as exc:  # noqa: BLE001
        raise LLMAnalysisError(f"LLM API call failed: {exc}") from exc

    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )

    try:
        parsed = _extract_json(raw_text)
    except (json.JSONDecodeError, AttributeError) as exc:
        raise LLMAnalysisError(
            f"Could not parse LLM response as JSON: {exc}\nRaw response:\n{raw_text}"
        ) from exc

    for key in ("strengths", "weaknesses", "recommendations", "interview_questions"):
        parsed.setdefault(key, [])

    return parsed


def fallback_analysis(matched_skills: List[str], missing_skills: List[str]) -> Dict:
    """A minimal, non-LLM fallback so the app still shows something useful
    if no API key is configured or the call fails."""
    return {
        "strengths": (
            [f"Resume already covers: {', '.join(matched_skills[:5])}"]
            if matched_skills
            else []
        ),
        "weaknesses": (
            [f"Resume is missing: {', '.join(missing_skills[:5])}"]
            if missing_skills
            else []
        ),
        "recommendations": [
            "Add an API key in the sidebar to get AI-generated, "
            "tailored recommendations and interview questions."
        ],
        "interview_questions": [],
    }
