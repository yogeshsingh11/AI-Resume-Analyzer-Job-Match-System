"""
nlp_processor.py
----------------
Text processing layer: regex-based contact extraction, spaCy-based
tokenization/lemmatization/NER, and keyword-based skill extraction.

Pipeline position:
    Resume Text / Job Description -> Regex + spaCy -> structured fields
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

import spacy
from spacy.language import Language

_NLP: Language | None = None

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?){2,4}\d{3,4}"
)
YEARS_EXP_RE = re.compile(
    r"(\d+)\+?\s*(?:years|yrs)\s*(?:of)?\s*experience", re.IGNORECASE
)
EDUCATION_KEYWORDS = [
    "b.tech", "btech", "bachelor", "b.sc", "bsc", "b.e", "be ",
    "m.tech", "mtech", "master", "m.sc", "msc", "mba", "phd",
    "ph.d", "diploma", "b.com", "bca", "mca",
]

# A reasonably broad taxonomy of common tech/professional skills.
# Extend this list as needed for the target job domain.
SKILLS_DB: List[str] = [
    # Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust",
    "ruby", "php", "sql", "r", "scala", "kotlin", "swift", "matlab",
    # Web / frameworks
    "react", "angular", "vue", "django", "flask", "fastapi", "node.js",
    "express", "streamlit", "next.js", "spring", "spring boot", ".net",
    # Data / ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "keras", "sentence transformers", "spacy", "opencv",
    "huggingface", "transformers", "llm", "generative ai", "prompt engineering",
    "data analysis", "data visualization", "statistics", "a/b testing",
    "matplotlib", "seaborn", "plotly", "tableau", "power bi",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "jenkins", "git", "github", "gitlab", "linux", "bash", "nginx",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "sqlite", "elasticsearch",
    "cassandra", "oracle", "dynamodb", "faiss", "chromadb", "pinecone",
    # Other tools / concepts
    "rest api", "graphql", "microservices", "agile", "scrum", "jira",
    "excel", "vba", "airflow", "spark", "hadoop", "kafka", "etl",
]


def _get_nlp() -> Language:
    """Lazily load the spaCy pipeline (loaded once per process)."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run:\n"
                "    python -m spacy download en_core_web_sm"
            ) from exc
    return _NLP


def extract_email(text: str) -> str | None:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    match = PHONE_RE.search(text)
    if not match:
        return None
    candidate = match.group(0).strip()
    digits = re.sub(r"\D", "", candidate)
    # Guard against matching random numeric noise (page numbers, years, etc.)
    if len(digits) < 7:
        return None
    return candidate


def extract_years_of_experience(text: str) -> int | None:
    matches = [int(m.group(1)) for m in YEARS_EXP_RE.finditer(text)]
    return max(matches) if matches else None


def extract_education(text: str) -> List[str]:
    lower = text.lower()
    found = sorted({kw.strip(".") for kw in EDUCATION_KEYWORDS if kw in lower})
    return found


def extract_organizations(text: str) -> List[str]:
    """Use spaCy NER to pull organization names (e.g. past employers)."""
    nlp = _get_nlp()
    doc = nlp(text[:100_000])  # guard against pathologically long input
    orgs = []
    seen: Set[str] = set()
    for ent in doc.ents:
        if ent.label_ == "ORG" and ent.text.strip().lower() not in seen:
            seen.add(ent.text.strip().lower())
            orgs.append(ent.text.strip())
    return orgs


def extract_skills(text: str) -> List[str]:
    """
    Keyword-based skill extraction against SKILLS_DB, using spaCy
    lemmatization so minor inflections still match.
    """
    nlp = _get_nlp()
    doc = nlp(text.lower()[:200_000])
    lemma_text = " ".join(tok.lemma_ for tok in doc)
    # Also keep the raw lowercase text so multi-word/skill-with-punctuation
    # entries like "c++", "ci/cd", "node.js" still match correctly.
    raw_text = " " + text.lower() + " "

    found = set()
    for skill in SKILLS_DB:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, raw_text) or re.search(pattern, lemma_text):
            found.add(skill)
    return sorted(found)


def process_document(text: str) -> Dict:
    """Run the full NLP extraction pipeline on a block of text."""
    return {
        "email": extract_email(text),
        "phone": extract_phone(text),
        "years_experience": extract_years_of_experience(text),
        "education": extract_education(text),
        "organizations": extract_organizations(text),
        "skills": extract_skills(text),
    }
