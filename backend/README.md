# PDF Agent Backend

A FastAPI service that ingests a PDF, builds a Chroma vector store with Ollama embeddings, and answers questions over it using a local Ollama LLM. Streaming responses (SSE), per-PDF dedup via content hash, and multi-turn chat history are supported.

## Project layout

- `backend/app.py` — FastAPI app: `/upload`, `/start_session`, `/query`, `/query/stream`, `/status`.
- `backend/vector.py` — builds (or reuses) a Chroma vector store from a PDF and returns a retriever.
- `backend/main.py` — optional CLI loop for asking questions against a PDF without the HTTP layer.
- `backend/uploaded_files/` — uploaded PDFs and per-PDF Chroma persistence dirs (auto-created).
- `backend/requirements.txt` — Python dependencies.

## Prerequisites

1. **Ollama** running locally (default: `http://127.0.0.1:11434`). Install from <https://ollama.com>.
2. Pull the models used by the project:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

3. Confirm Ollama is up:

```bash
ollama list
```

## Setup (macOS / zsh)

1. Create and activate a virtualenv:

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

## Run the API server

From the `backend/` directory:

```bash
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

On startup the service performs a one-shot warmup call against Ollama so the first user query doesn't pay the model-load cost. The LLM and embedding model are configured with `keep_alive=-1` so they stay resident across requests.

## Run the CLI (optional)

`main.py` runs an interactive Q&A loop without HTTP:

```bash
cd backend
python main.py
```

It looks for `Employee-Handbook.pdf` next to `main.py`; if that file is absent, it prompts for a PDF path.

## Endpoints

- `POST /upload` — multipart upload (field `file`). Returns `{doc_id, filename, took_seconds, chunk_count, reused}`. The `reused` flag is `true` when the same PDF (by SHA-256) was previously indexed; the existing `doc_id` is returned and no re-embedding happens.
- `POST /start_session?doc_id=<doc_id>` — opens a chat session whose history is fed back into subsequent prompts. Returns `{session_id}`.
- `POST /query` — JSON `{question, session_id?, doc_id?}`. Synchronous; returns `{answer, doc_id, debug}`.
- `POST /query/stream` — same payload as `/query`. Returns `text/event-stream` SSE with `event: token` deltas and a final `event: done` carrying `{doc_id, debug}`. Used by the frontend.
- `GET /status` — uploaded doc ids and active sessions.

## Example curl flows

Upload a PDF:

```bash
curl -F "file=@/path/to/your.pdf" http://127.0.0.1:8000/upload
```

Start a session:

```bash
curl -X POST "http://127.0.0.1:8000/start_session?doc_id=<doc_id>"
```

Ask a question (synchronous):

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<session_id>","question":"What is the recruitment process?"}'
```

Ask a question (streaming):

```bash
curl -N -X POST "http://127.0.0.1:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<session_id>","question":"Summarize the leave policy"}'
```

Or query directly by `doc_id` (no session, no history):

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"<doc_id>","question":"What is the recruitment process?"}'
```

## How indexing & dedup work

- On upload the server hashes the PDF bytes (SHA-256, first 16 hex chars) and uses that as both the on-disk filename prefix and the Chroma `persist_directory` name (`uploaded_files/<hash>_chroma`).
- If the persist dir already contains embeddings, they are reused — re-uploading the same PDF is effectively free.
- An in-memory `HASH_INDEX` also returns the existing `doc_id` for an exact re-upload during the same process lifetime.

## Configuration knobs (in `app.py`)

- `OllamaLLM(model="llama3.2", temperature=0, num_ctx=8192, num_predict=512, keep_alive=-1)`
- `OllamaEmbeddings(model="mxbai-embed-large", keep_alive=-1)` (in `vector.py`)
- Chunking: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` in `vector.py`.
- Retrieval: similarity search with `k=5`.

## Rebuilding a vector store

Delete the per-PDF persist dir to force a re-embed on next upload:

```bash
rm -rf backend/uploaded_files/<hash>_chroma
```

## Troubleshooting

- **`ConnectionError` to `127.0.0.1:11434`** — Ollama isn't running. Start it (`ollama serve` or launch the app) and confirm `ollama list` works.
- **Slow first request** — the warmup at startup mitigates this, but if it failed (Ollama not yet up), the first real query will pay the model-load cost.
- **`ModuleNotFoundError`** — activate the venv (`source backend/.venv/bin/activate`) and re-run `pip install -r backend/requirements.txt`.
- **Empty/garbage answers** — check `/query` debug payload; if `retrieved_count` is 0 the keyword-overlap fallback kicks in. Verify the PDF text-extracted correctly (scanned PDFs need OCR upstream).
