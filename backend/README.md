# PDF Agent Backend

This backend builds a small vector store from an Employee Handbook PDF and exposes a retriever for semantic search. It's a minimal example using the Ollama embeddings and Chroma vector store.

## Project layout

- `backend/main.py` - example runner that imports the `retriever` from `vector.py` and runs queries.
- `backend/vector.py` - builds the vector store at import-time from `Employee-Handbook.pdf`; creates `retriever`.
- `backend/Employee-Handbook.pdf` - the source PDF used to build the vector DB.
- `backend/pdf_chroma_db/` - persisted Chroma DB directory (created after building vectors).
- `backend/requirements.txt` - Python dependencies for the project.

## Quick setup (macOS / zsh)

1. Create a virtual environment and activate it (recommended):

```bash
python3 -m venv backend/venv
source backend/venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

3. Run the FastAPI service (this project now exposes a small API):

```bash
# From the backend/ directory
uvicorn app:app --reload --port 8000
```

Endpoints:

- POST /upload - multipart file upload (field name: `file`) to upload a PDF and build a vector DB. Returns `doc_id`.
- POST /start_session - JSON body {"doc_id": "..."} returns `session_id` to continue a conversation.
- POST /query - JSON body {"question": "...", "session_id": "..."} or {"question": "...", "doc_id": "..."} to ask a question.
- GET /status - returns uploaded doc ids and active sessions.

Example curl flows:

Upload a PDF:

```bash
curl -F "file=@/path/to/Employee-Handbook.pdf" http://127.0.0.1:8000/upload
```

Start a session:

```bash
curl -X POST "http://127.0.0.1:8000/start_session" -H "Content-Type: application/json" -d '{"doc_id":"<doc_id>"}'
```

Ask a question (using session):

```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"session_id":"<session_id>","question":"What is the recruitment process?"}'
```

Or ask a question directly by doc_id:

```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"doc_id":"<doc_id>","question":"What is the recruitment process?"}'
```

Notes:
- The project resolves the PDF path relative to `vector.py`. If you move files, ensure `Employee-Handbook.pdf` is next to `vector.py` or update the path.
- The first import of `vector.py` will build the vector DB and may take some time and memory. The code checks for the `pdf_chroma_db` directory and only re-adds documents if it doesn't exist.

## Troubleshooting

- ValueError: "File path Employee_Handbook.pdf is not a valid file or url"
  - This happens if the loader receives the wrong filename or a relative path that doesn't point to the PDF. `vector.py` resolves the path relative to the file using `os.path.join(os.path.dirname(__file__), "Employee-Handbook.pdf")`.
  - Confirm the file exists next to `vector.py`:

```bash
ls -l backend/Employee-Handbook.pdf
```

- Missing packages / ModuleNotFoundError
  - Activate the virtualenv and run `pip install -r backend/requirements.txt`. If errors persist, check the Python version (this project used Python 3.13 when created).

## Optional improvements

- Make vector DB creation lazy: move heavy initialization from module scope into a function like `get_retriever()` so importing `vector` is lightweight. This prevents long import times and side effects during unit tests.

- Add a small CLI to build or query the DB explicitly instead of building on import.

## Rebuilding the vector DB

To force-rebuild the vector DB, remove the `backend/pdf_chroma_db/` directory (or back it up) and run `python backend/main.py` again.

```bash
rm -rf backend/pdf_chroma_db
python backend/main.py
```

## Contact

If you need help with packages (e.g., `langchain_ollama`, `langchain_chroma`), include the pip install output and the Python version in your issue.