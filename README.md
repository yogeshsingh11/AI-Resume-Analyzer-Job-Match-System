# AI Resume Analyzer & Job Match System

A Streamlit-based application that analyzes a resume PDF against a job description and provides a resume–job match score, skill-gap analysis, strengths and weaknesses, improvement suggestions, and tailored interview questions.

## Features

- Extracts text from resume PDFs using **PyMuPDF**
- Processes resume and job-description text using **Regex and spaCy**
- Extracts structured information and relevant skills
- Generates semantic embeddings using **Sentence Transformers**
- Calculates resume–job similarity using **cosine similarity**
- Produces a **match score**
- Identifies **skill gaps**
- Uses the **Anthropic Claude API** for qualitative analysis
- Generates:
  - Strengths
  - Weaknesses
  - Improvement recommendations
  - Tailored interview questions
- Provides an interactive **Streamlit dashboard**
- Can calculate the match score and skill-gap analysis locally without an API key

## How It Works

```text
Resume PDF ──> PyMuPDF ──> Text
                              │
                              ▼
                    Regex + spaCy Processing
                              │
Job Description ──────────────┘
                              │
                              ▼
                  Sentence Transformer
                    Semantic Embeddings
                              │
                              ▼
                    Cosine Similarity
                              │
                              ▼
                       Match Score
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Skill Gap Analysis    Claude LLM
                                        │
                                        ▼
                         Strengths / Weaknesses /
                         Recommendations /
                         Interview Questions
                              │
                              ▼
                       Streamlit Dashboard
```

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **PyMuPDF** | Resume PDF text extraction |
| **Regex** | Pattern-based information and skill extraction |
| **spaCy** | NLP processing |
| **Sentence Transformers** | Semantic embeddings |
| **Cosine Similarity** | Resume–job similarity calculation |
| **Anthropic Claude API** | AI-based qualitative analysis |
| **Prompt Engineering** | Structuring LLM analysis and outputs |
| **Streamlit** | Web application interface |
| **Pandas / NumPy** | Data processing and numerical operations |

## Project Structure

```text
AI-Resume-Analyzer-Job-Match-System/
│
├── app.py                  # Streamlit UI and application orchestration
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Files ignored by Git
│
└── src/
    ├── __init__.py
    ├── pdf_parser.py       # PDF text extraction using PyMuPDF
    ├── nlp_processor.py    # Regex + spaCy processing and skill extraction
    ├── matcher.py          # Embeddings, similarity score and skill-gap analysis
    └── llm_analyzer.py     # Anthropic API calls and prompt engineering
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yogeshsingh11/AI-Resume-Analyzer-Job-Match-System.git
cd AI-Resume-Analyzer-Job-Match-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

The project uses the `en_core_web_sm` model for NLP processing.

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure the Anthropic API key

The application uses an Anthropic API key for the qualitative AI analysis.

You can obtain an API key from the Anthropic Console:

https://console.anthropic.com/

Enter the key in the application's sidebar when prompted.

**Do not commit API keys, passwords, or other secrets to GitHub.**

## Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, typically:

```text
http://localhost:8501
```

### Using the Application

1. Enter your Anthropic API key in the sidebar.
2. Upload your resume as a PDF.
3. Paste the job description.
4. Click **Analyze Resume**.
5. Review the match score, skill gaps, recommendations, and interview questions.

If no API key is provided, the application can still perform the local match-score and skill-gap analysis, while the full LLM-based qualitative analysis is unavailable.

## First Run Notes

- The Sentence Transformer model (`all-MiniLM-L6-v2`) and spaCy model may download the first time they are used.
- These models are cached locally, so later runs can be faster.
- The resume should contain selectable text. If the PDF is only a scanned image, PyMuPDF may not be able to extract its text.

## Limitations

- Scanned/image-only PDFs may not produce usable text without OCR.
- The quality of the analysis depends on the quality and completeness of the resume and job description.
- LLM-generated recommendations depend on the Anthropic API response.
- The current version is designed for individual resume–job-description analysis rather than managing a large candidate database.

## Future Improvements

Possible extensions for future versions include:

- Adding OCR support for scanned resumes
- Storing previous analyses and match scores
- Supporting multiple resumes and candidate comparison
- Adding richer visual dashboards
- Improving skill extraction with a larger and more domain-specific skill taxonomy
- Adding a vector database for searching across multiple candidate profiles

## Project Pipeline

```text
PDF
  ↓
Text Extraction
  ↓
NLP Processing
  ↓
Semantic Embeddings
  ↓
Cosine Similarity
  ↓
Match Score
  ↓
Skill Gap Analysis
  ↓
LLM Analysis
  ↓
Recommendations & Interview Questions
  ↓
Streamlit Dashboard
```
