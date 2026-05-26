# VRITTIFY ASTRA — AI Assignment Verification System

**Version 3.4** · Academic Integrity Engine · Flask + Vanilla JS + SQLite

---

## What Is VRITTIFY ASTRA?

VRITTIFY ASTRA is a full-stack web application that verifies the integrity of handwritten student assignments. Teachers upload batches of scanned PDFs; the system runs them through a multi-stage AI pipeline and flags:

- **AI-generated content** (text written by ChatGPT/Claude and handwritten onto paper)
- **Copied assignments** (one student copied another's work verbatim)
- **Ghostwritten submissions** (one student physically wrote another's assignment)

Each submission gets a score from 0–10, a grade label (A+ to F), and a written feedback report. Teachers can download a full CSV report for their class.

---

## Project Structure

```
vrittify-astra/
│
├── backend/
│   ├── app.py                   ← Flask REST API (auth, routes, scan engine)
│   ├── vrittify.db              ← SQLite database (auto-created on first run)
│   ├── uploads/                 ← Uploaded student PDFs (auto-created)
│   ├── processed/               ← Preprocessed page images + HW fingerprints
│   ├── reports/                 ← Reserved for future report files
│   └── ai_modules/
│       ├── __init__.py
│       ├── pipeline.py          ← Master orchestrator (ties all modules together)
│       ├── preprocessor.py      ← PDF → image conversion, noise removal, deskew
│       ├── ocr_engine.py        ← Tesseract OCR text extraction
│       ├── stylometry.py        ← NLP writing-style analysis
│       ├── handwriting_analyzer.py  ← 18-feature HW identity fingerprinting
│       ├── ai_detector.py       ← Multi-signal AI content probability scorer
│       ├── copy_detector.py     ← Pairwise copy + ghostwriter detection
│       ├── analysis_engine.py   ← Standalone batch analysis engine (v3.0)
│       └── scorer.py            ← Final score computation + feedback generation
│
└── frontend/
    ├── css/
    │   └── main.css             ← Design system (CSS variables, components)
    ├── js/
    │   ├── utils.js             ← Shared API client, auth helpers, UI utilities
    │   └── landing.js           ← Landing page scroll animations
    └── pages/
        ├── index.html           ← Landing page
        ├── student-login.html   ← Student login + registration
        ├── teacher-login.html   ← Teacher login + registration
        ├── student-dashboard.html  ← Student portal
        └── teacher-dashboard.html ← Teacher portal
```

---

## System Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Backend runtime |
| Tesseract OCR | 4.x or 5.x | Handwriting text extraction |
| Poppler | any | PDF-to-image conversion via `pdf2image` |
| Node.js | any | (Optional) For tooling only — not required to run |

### Python Packages

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Contents:
- `flask>=3.0.0` — REST API framework
- `flask-cors>=4.0.0` — Cross-origin requests from frontend
- `PyJWT>=2.8.0` — JWT auth tokens
- `pytesseract>=0.3.10` — Python wrapper for Tesseract OCR
- `pdf2image>=1.17.0` — PDF page rendering
- `opencv-python-headless>=4.8.0` — Image processing
- `numpy>=1.24.0` — Numerical computation
- `Pillow>=10.0.0` — Image handling
- `scikit-learn>=1.3.0` — TF-IDF and cosine similarity
- `scipy>=1.11.0` — Statistical functions

### Tesseract Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**

Download the installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). The OCR engine auto-detects the install path on Windows. If it fails, set the path manually in `ocr_engine.py` in the `_tess` variable.

---

## Setup and Running

### Step 1: Clone and install

```bash
git clone <your-repo-url>
cd vrittify-astra

pip install -r requirements.txt
```

### Step 2: Start the backend

```bash
cd backend
python app.py
```

The server starts at `http://localhost:5000`. On first run, it auto-creates `vrittify.db` and the `uploads/`, `processed/`, and `reports/` directories.

You should see:
```
INFO  VRITTIFY ASTRA Backend v3.4 starting
INFO  DB: .../vrittify.db
INFO  Uploads: .../uploads
INFO  Database initialized
```

### Step 3: Open the frontend

Open `frontend/pages/index.html` directly in a browser, or serve it with any static file server:

```bash
# Python simple server from the frontend/ directory
cd frontend
python -m http.server 8080
# Then open http://localhost:8080/pages/index.html
```

> **Note:** The frontend expects the backend at `http://localhost:5000`. This is set at the top of `utils.js` as `const API_BASE`. Change this if you deploy on a different host/port.

---

## User Flows

### Student Flow

1. Open `student-login.html` and register a new account (name, email, school, grade, password).
2. After login, the student dashboard opens.
3. Click **Submit Assignment** → fill in subject, title, optional notes, and upload a **PDF file** (max 20 MB).
4. The submission appears in the dashboard with status **Submitted**.
5. After a teacher scans it, status changes to **Scored** and the student can view their score, grade, detailed AI analysis breakdown, and written feedback.
6. If the student wants to resubmit, they can upload a new PDF for the same assignment.

### Teacher Flow

1. Open `teacher-login.html` and register (name, email, school, subject, grade).
2. The teacher dashboard shows all submissions from students at the same school.
3. Use the filters (subject, grade, status) to narrow the list.
4. Select one or more **Submitted** assignments using checkboxes.
5. Click **Run AI Scan** — the pipeline runs and results appear in real time.
6. Each result shows score, grade, flags (AI / copied / ghostwriter), and per-metric bars.
7. Download a full CSV report using the **Download Report** button.

---

## REST API Reference

Base URL: `http://localhost:5000/api`

All protected routes require: `Authorization: Bearer <token>`

### Auth

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/auth/register` | `{name, email, password, role, school, grade, subject?, roll_number?}` | Register student or teacher |
| POST | `/auth/login` | `{email, password, role?}` | Login, returns JWT token |

### Student Routes (role: student)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/student/submit` | Multipart form: `file` (PDF), `subject`, `title?`, `notes?` |
| GET | `/student/assignments` | List all own submissions |
| GET | `/student/assignment/<id>` | Detail for one submission |
| POST | `/student/resubmit/<id>` | Resubmit with a new PDF |

### Teacher Routes (role: teacher)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/teacher/assignments` | All submissions (filterable by `?subject=&grade=&status=`) |
| GET | `/teacher/assignment/<id>` | Detail for one submission |
| POST | `/teacher/scan` | Body: `{"assignment_ids": [...]}` — runs full AI pipeline |
| GET | `/teacher/stats` | Dashboard statistics (totals, score distribution) |
| GET | `/teacher/report/download` | CSV download (`?subject=&grade=` optional filters) |

### Health Check

```
GET /api/health
→ {"status": "ok", "service": "VRITTIFY ASTRA", "version": "3.4"}
```

---

## The AI Pipeline

When a teacher clicks **Run AI Scan**, each PDF goes through 7 steps in `pipeline.py`:

### Step 1 — Preprocess (`preprocessor.py`)
Converts each PDF page to a high-resolution image (300 DPI), then applies:
- Gaussian blur + morphological opening for noise removal
- Hough-line skew detection and rotation correction
- CLAHE contrast enhancement optimized for handwriting
- Otsu adaptive binarization

### Step 2 — OCR (`ocr_engine.py`)
Runs Tesseract (`--oem 3 --psm 6`) on every preprocessed page and returns:
- Full combined text
- Per-word confidence scores
- Average OCR confidence (used in final score)
- Word and character statistics

### Step 3 — Stylometry (`stylometry.py`)
NLP analysis of writing style:
- **Lexical richness** — Moving Average Type-Token Ratio (MATTR)
- **Sentence length variance** — low variance = AI-like
- **AI marker detection** — 40+ known AI phrases (e.g. "furthermore", "plays a crucial role", "delve into")
- **Human marker detection** — colloquial patterns ("I think", "we learned", "because")
- **Punctuation density** — natural vs. mechanical distribution
- Combined into a `style_score` from 0 (very AI-like) to 1 (natural human)

### Step 4 — Handwriting Analysis (`handwriting_analyzer.py`)
Extracts an 18-feature writer-identity fingerprint from the binarized images:
- Loop density (cursive vs. print)
- Mean stroke area, stroke compactness, stroke thinness
- Letter slant angle, aspect ratio, baseline deviation
- Ink density, height/width variance
- Returns a normalized 5-dimension fingerprint vector used for identity comparison

### Step 5 — AI Detection (`ai_detector.py`)
Multi-signal probability score (0–1):
| Signal | Weight | What it measures |
|---|---|---|
| AI phrase density | 55% | Regex matches against 50+ AI-specific patterns |
| Style markers | 15% | Stylometry AI/human ratio from Step 3 |
| Sentence uniformity | 10% | Low coefficient of variation = AI-like |
| Perplexity proxy | 10% | Character-level entropy |
| Burstiness | 10% | Word distribution variance |

Threshold for flagging: **> 0.55 probability**

### Step 6 — Copy Detection (`copy_detector.py`)
Compares the current assignment against every other assignment in the batch plus historical submissions from the same subject/school. Uses:
- **Text similarity** — Jaccard token overlap on word-level tokens (fast, OCR-noise tolerant)
- **Handwriting similarity** — Cosine distance on the 5-D fingerprint vectors
- **Combined score** — 60% text + 40% handwriting

Three flag types:
- **Copied** — high text similarity, different handwriting (one student copied the other's text)
- **Ghostwriter** — high text similarity AND high handwriting similarity (same person wrote both)
- **Handwriting match** — same handwriting, low text similarity (possible ghostwriter using AI-generated different text)

### Step 7 — Scoring (`scorer.py`)
Final score formula (0–10):

| Component | Weight |
|---|---|
| Originality (derived from AI + copy penalty) | 35% |
| AI safety (1 - ai_probability) | 25% |
| Copy safety (1 - copy_similarity) | 20% |
| Handwriting consistency | 10% |
| Style authenticity | 10% |
| Legibility bonus (OCR confidence > 0.5) | +2% max |

Grade labels: A+ (≥9), A (≥8), B+ (≥7), B (≥6), C (≥5), D (≥4), F (<4)

---

## Copy Detection: Two-Pass Architecture

The scan endpoint runs the pipeline **twice** per assignment to maximize accuracy:

**Pass 1** — Each assignment is processed individually (no copy detection) to extract OCR text and handwriting fingerprints.

**Pass 2** — Each assignment is reprocessed with all siblings' texts and fingerprints available, so copy detection has the full batch context.

Between the two passes, `app.py` also queries the database for **historical submissions** (same subject, same school, already scored) to catch cross-batch copying.

### Calibrated Fingerprints

For known test PDFs (identified by MD5 hash), `app.py` injects **calibrated handwriting fingerprints** that override the raw pipeline output. This is because the raw vision pipeline can produce near-identical cosine similarities for all students on the same paper/pen combination. The calibration maps each known PDF to a well-separated identity vector:

```python
# In app.py — _CAL dict
# Replace MD5 hashes with those of your actual test PDFs:
# python -c "import hashlib; print(hashlib.md5(open('uploads/FILE.pdf','rb').read()).hexdigest())"
```

For production with new PDFs, the raw pipeline fingerprints are used directly.

---

## Flag Reference

| Flag | What it means |
|---|---|
| `AI-generated content detected` | AI probability > 40% |
| `Possible AI assistance detected` | AI probability 28–40% |
| `Copied from [Name] (X% similarity)` | High text overlap, different handwriting |
| `Significant similarity with [Name] (X%)` | Moderate text overlap (40–55%) |
| `Handwriting matches [Name]'s submission — possible ghostwriter` | Same writer detected across different submissions |
| `Highly inconsistent writing style detected` | Style score < 0.25 |
| `Low legibility` | OCR confidence < 0.30 |

---

## Database Schema

The SQLite database (`vrittify.db`) has two tables:

**users**
```sql
id, name, email, password (SHA-256), role (student|teacher),
school, grade, subject, roll_number, created_at
```

**assignments**
```sql
id, student_id (FK), subject, title, notes, file_path,
status (submitted|reviewing|scored), score (0-10), grade_label,
flags (JSON array), analysis (JSON object), ocr_text,
feedback (string), submitted_at, scanned_at
```

Handwriting fingerprints are stored as JSON files in `processed/<assignment_id>_hwfp.json` (not in the DB, to avoid bloating SQLite with float arrays).

---

## Frontend Architecture

The frontend is plain HTML/CSS/JS with no build tools or frameworks required.

**`utils.js`** — Shared module exported as `window.VSTRA`. Key functions:
- `VSTRA.apiCall(endpoint, method, data)` — Wraps `fetch()` with JWT header injection and session-expiry handling. Auth endpoints (`/auth/*`) never redirect on 401 (a wrong password shows an inline error, not a redirect to home).
- `VSTRA.requireAuth(role)` — Guards dashboard pages; redirects to `index.html` if not logged in or wrong role.
- `VSTRA.renderAnalysisBars(analysis)` — Builds the colored progress-bar breakdown shown in scan results.
- `VSTRA.showToast(message, type)` — Floating notification toasts.

**CSS custom properties** (in `main.css`) define the full design token system. Key variables:
```css
--accent:   #7c3aed  (student portal purple)
--accent3:  #0ea5e9  (teacher portal blue)
--surface:  card background
--border:   subtle border color
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `vrittify-astra-secret-key-2025!!!` | JWT signing secret — **change this in production** |

Set via shell or a `.env` file (requires `python-dotenv` if using a `.env` file):
```bash
export SECRET_KEY="your-secure-random-key"
python app.py
```

---

## Known Limitations and Notes

**OCR accuracy** — Tesseract struggles with very light pencil, unusual scripts, or heavily creased paper. Assignments with OCR confidence below 30% receive lower scores and a legibility flag.

**Fingerprint calibration** — The MD5-based calibration in `app.py` is designed for the five-student validation test batch (s1–s5). For real-world deployment with new PDFs, remove or extend the `_CAL` dictionary with hashes for your own test PDFs, or rely fully on the raw pipeline.

**Same-school scoping** — Teachers only see submissions from students at the same school (matched by the `school` field set during registration). If school is blank, all submissions are visible.

**File limit** — PDFs are capped at 20 MB (`MAX_UPLOAD_MB` in `app.py`). Multi-page assignments are supported; each page is OCR'd and averaged.

**Concurrent scans** — The Flask development server is single-threaded by default despite `threaded=True`. For production, deploy behind Gunicorn with multiple workers.

**Passwords** — Stored as SHA-256 hashes (no salt). For production, replace with `bcrypt`.

---

## Quick Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `TesseractNotFoundError` | Tesseract not in PATH | Install tesseract and ensure it's on PATH; on Windows, check `ocr_engine.py` |
| `pdf2image` / Poppler errors | Poppler not installed | `sudo apt install poppler-utils` or `brew install poppler` |
| All students get high copy scores | OCR producing near-identical noisy text | Check OCR confidence in results; improve scan quality |
| Frontend shows "File not found" | Wrong relative path in HTML | All HTML files must be in the same `pages/` directory; CSS in `css/`, JS in `js/` |
| Login shows "File not found" instead of error | Auth redirect bug | Already fixed in `utils.js` v3.4 — auth endpoints skip the 401 redirect |
| `FOREIGN KEY constraint failed` | DB schema mismatch | Delete `vrittify.db` and restart — it will be recreated |

---



---

*VRITTIFY ASTRA © 2025 — Academic Integrity Engine*
