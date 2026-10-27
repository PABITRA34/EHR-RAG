# EHRLoom

EHRLoom is a prototype clinical-history question-answering application. It retrieves patient-scoped encounter records from PostgreSQL, uses pgvector semantic search for open-ended history questions, redacts selected personal identifiers, and routes factual questions through NeMo Guardrails to a configured language model.

> **Important:** This is a research/demo project, not a medical device or clinical decision-support system. It may produce incorrect or incomplete answers. Do not use it to diagnose, prescribe, or make treatment decisions. Always consult a qualified clinician.

## Features

- FastAPI backend with patient-history chat and a patient list.
- Streamlit interface for selecting a patient and asking history questions.
- Targeted retrieval for admission, medication, and report questions, with semantic retrieval for other history questions.
- PostgreSQL storage with 768-dimensional pgvector embeddings.
- Presidio-based redaction of selected identifiers before retrieved context and questions are sent to the configured LLM.
- A chat policy that blocks requests for diagnosis or treatment recommendations.

## Project layout

```text
data/                       Local input data (not committed)
scripts/
  ingest_baseline_data.py   Create the base table and append CSV rows
  verify_ingestion.py       Check row count and display sample fields
  apply_vector_schema.py    Add the 768-dimensional embedding column
  generate_embeddings.py    Generate embeddings for up to 100 records
  test_vector_search.py     Run a sample pgvector similarity search
  test_guardrails.py        Exercise valid and blocked guardrail prompts
src/
  api/                      FastAPI application and chat policy
  database/schema.sql       Base patient_encounters table
  guardrails/config.yml     LLM and embedding model configuration
  pii_redaction/            Presidio-based redaction
  ui/                       Streamlit application
tests/
  test_chat_policy.py       Unit tests for chat policy and retrieval modes
```

## Screenshots

<p align="center">
  <img src="./WhatsApp%20Image%202026-10-27%20at%2011.43.58%20PM.jpeg" alt="EHRLoom screenshot 1" width="45%" />
  <img src="./WhatsApp%20Image%202026-10-27%20at%2011.45.33%20PM.jpeg" alt="EHRLoom screenshot 2" width="45%" />
</p>

## Requirements

- Python version compatible with the pinned packages in `requirements.txt`.
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension installed and enabled.
- A dataset CSV at `data/MIMIC_IV_Trasncript.csv` with columns matching the table schema in `src/database/schema.sql`.
- A valid Gemini API key for the model configured in `src/guardrails/config.yml`.

The application loads the BioClinical ModernBERT embedding model from Hugging Face when it starts or when an embedding script runs. The first run may download model files.

## Setup

Run these commands from the project root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` file in the project root. Do not commit it:

```dotenv
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=clinical_db
GEMINI_API_KEY=your_gemini_api_key
```

The database user needs permission to create and modify the application table. Enable pgvector in the target database before running the vector-schema script:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The extension may need to be installed by a database administrator, depending on your PostgreSQL hosting provider.

## Prepare the database

Place the authorized input dataset at `data/MIMIC_IV_Trasncript.csv`, then run the scripts in order:

```powershell
python .\scripts\ingest_baseline_data.py
python .\scripts\verify_ingestion.py
python .\scripts\apply_vector_schema.py
python .\scripts\generate_embeddings.py
```

The ingestion script creates the base table from `src/database/schema.sql` and **appends** the CSV rows. It is not an idempotent import: running it again can create duplicate rows. The embedding script processes at most 100 records per run and prints clinical data during verification/search scripts; use it only in an appropriately protected environment.

To try the sample vector search after embeddings have been generated:

```powershell
python .\scripts\test_vector_search.py
```

To exercise the configured guardrails:

```powershell
python .\scripts\test_guardrails.py
```

## Run the application

Start the API from the project root:

```powershell
uvicorn src.api.main:app --reload
```

The API listens on `http://localhost:8000`. Start the Streamlit UI in a second terminal:

```powershell
streamlit run .\src\ui\app.py
```

The UI connects to the API at `http://localhost:8000`. The API initializes the redactor, guardrails, and embedding model during startup.

## Run unit tests

```powershell
python -m unittest discover -s tests -v
```

## Data and privacy

Clinical and patient-linked data is sensitive. Use only data you are authorized to process, follow the applicable dataset terms and privacy requirements, and restrict access to the database, environment variables, logs, and machine running the application. Redaction reduces some exposure but does not guarantee that all identifying information is removed.

The repository ignores `.env` files and `data/*.csv`. Keep credentials and datasets out of commits, including private repositories, unless you have explicitly verified that sharing is permitted. The dataset source notes, if applicable, are in `data/data.txt`.