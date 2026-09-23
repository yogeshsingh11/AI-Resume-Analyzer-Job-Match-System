# 📄 AI Resume Analyzer & Job Match System

A Streamlit web app that analyzes a resume PDF against a job description and
produces a match score, skill-gap breakdown, strengths/weaknesses,
improvement suggestions, and tailored interview questions.

## How it works

```
Resume.pdf ─▶ PyMuPDF ─▶ Text ─▶ Regex + spaCy ─▶ structured fields
                                                        │
Job Description ─────────────────────▶ Regex + spaCy ──┘
                                                        │
                                        Sentence Transformer (embeddings)
                                                        │
                                              Cosine Similarity
                                                        │
                                                 Match Score (%)
                                                        │
                                    ┌───────────────────┴───────────────────┐
                              Skill Gap Analysis                   LLM Analysis (Claude)
                                                                     │
                                                          Strengths / Weaknesses /
                                                          Recommendations / Interview Qs
                                                                     │
                                                          Streamlit Dashboard
```

## Tech stack

Python · Pandas · NumPy · PyMuPDF · Regex · spaCy · Sentence Transformers ·
Cosine Similarity · Anthropic Claude API · Prompt Engineering · Streamlit

## Project structure

```
AI-Resume-Analyzer/
├── app.py                  # Streamlit UI and orchestration
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── pdf_parser.py        # PyMuPDF text extraction
│   ├── nlp_processor.py     # Regex + spaCy extraction, skill taxonomy
│   ├── matcher.py           # Embeddings, cosine similarity, skill gap
│   └── llm_analyzer.py      # Anthropic API calls + prompt engineering
└── assets/
    └── screenshots/
```

## Setup

1. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the spaCy language model** (used for tokenization, lemmatization,
   and named entity recognition):

   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Get an Anthropic API key** (used for the qualitative analysis — strengths,
   weaknesses, recommendations, interview questions):

   - Sign up / log in at [console.anthropic.com](https://console.anthropic.com)
   - Create an API key
   - You'll paste this into the app's sidebar at runtime — it is never
     written to disk. (Alternatively, set it as the `ANTHROPIC_API_KEY`
     environment variable and adapt `app.py`'s sidebar default if you'd
     rather not paste it each time.)

## Running the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

1. Paste your Anthropic API key into the sidebar.
2. Upload a resume PDF.
3. Paste a job description.
4. Click **Analyze Resume**.

Without an API key, the app still runs end-to-end — the match score and
skill-gap analysis are computed locally — but strengths/weaknesses,
recommendations, and interview questions fall back to a minimal
non-AI summary instead of full LLM output.

## Notes on the first run

- The Sentence Transformer model (`all-MiniLM-L6-v2`) and the spaCy model
  download automatically / on first use and are cached locally afterward,
  so the very first analysis will be slower than subsequent ones.
- If your resume PDF is a scanned image rather than real text, PyMuPDF
  won't be able to extract text from it — export a text-based PDF instead
  (e.g. "Print to PDF" from a Word doc / Google Doc).

## Extending the project

Ideas for going further once the base version works end-to-end:

- **FAISS / ChromaDB** — store embeddings for multiple resumes and search
  across a candidate pool.
- **SQLite** — persist past analyses, match scores, and job descriptions.
- **Plotly** — richer dashboards (radar charts, skill-coverage gauges).
- **LangChain** — only if you outgrow direct API calls; not required here.

## The interview-ready summary

If asked "what technologies did you use?":

> Python, Pandas, NumPy, PyMuPDF, Regex, spaCy, Sentence Transformers,
> semantic embeddings, cosine similarity, the Anthropic Claude API,
> prompt engineering, Streamlit, and Git/GitHub.

Pipeline to memorize:

> PDF → Text Extraction → NLP → Embeddings → Semantic Similarity →
> Match Score → Skill Gap Analysis → LLM → Recommendations → Streamlit
