# PDF Agent

Chat with any PDF, locally. Upload a document, ask questions about it in natural language, and stream answers back token-by-token. Everything runs on your machine — the LLM and embeddings are served by [Ollama](https://ollama.com), so no document content leaves the host.

## What it does

- **Ingests a PDF** — parses pages, splits into ~1000-char chunks (with overlap), embeds them with `mxbai-embed-large`, and stores the vectors in a per-PDF Chroma database on disk.
- **Answers questions** — retrieves the top-5 most relevant chunks for your question and asks `llama3.2` to answer using only that context.
- **Streams responses** — tokens arrive over Server-Sent Events; the UI renders them as they're produced.
- **Remembers the conversation** — sessions inject the last few turns back into the prompt so follow-up questions have context.
- **Dedupes uploads** — identical PDFs (matched by SHA-256) reuse the existing vector store instead of re-embedding.

## Architecture

```
┌──────────────┐   PDF upload / SSE chat    ┌────────────────┐
│  React +     │ ─────────────────────────▶ │  FastAPI       │
│  Parcel UI   │                            │  (uvicorn)     │
│  :5173       │ ◀───────────────────────── │  :8000         │
└──────────────┘                            └────┬───────────┘
                                                 │
                                  ┌──────────────┼──────────────┐
                                  ▼              ▼              ▼
                              PyPDFLoader    Chroma DB    Ollama HTTP
                              + splitter     (per-PDF)    (:11434)
                                                          ├─ llama3.2
                                                          └─ mxbai-embed-large
```

- `backend/` — FastAPI service. Endpoints: `POST /upload`, `POST /start_session`, `POST /query`, `POST /query/stream`, `GET /status`.
- `frontend/` — React UI bundled with Parcel.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **Ollama** running locally (default `http://127.0.0.1:11434`)

Install Ollama from <https://ollama.com>, then pull the two models the app uses:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

Confirm Ollama is running:

```bash
ollama list
```

## Start the backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

On startup the service performs a one-shot warmup against Ollama so the first user query doesn't pay the model-load cost. The LLM and embedding model are configured with `keep_alive=-1` to stay resident across requests.

The API is now reachable at <http://127.0.0.1:8000>. See `backend/README.md` for endpoint details and curl examples.

## Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm start
```

Parcel serves the UI at <http://localhost:5173> with hot reload. By default it talks to the backend at `http://127.0.0.1:8000` — to point it elsewhere, edit the `<meta name="api-base" …>` tag in `frontend/public/index.html`.

## Use it

1. Open <http://localhost:5173>.
2. Upload a PDF. The first upload takes a few seconds (parse + embed); identical re-uploads are instant.
3. Ask questions in the chat box. The first question for a fresh upload silently opens a session so subsequent turns have history.

## Project layout

```
pdf-agent/
├── README.md              ← you are here
├── backend/
│   ├── app.py             ← FastAPI app (endpoints, streaming, sessions)
│   ├── vector.py          ← PDF loading + Chroma vector store
│   ├── main.py            ← optional CLI Q&A loop
│   ├── requirements.txt
│   ├── uploaded_files/    ← saved PDFs + per-PDF Chroma persist dirs
│   └── README.md
└── frontend/
    ├── src/
    │   ├── App.jsx        ← upload UI + streaming chat
    │   ├── main.jsx
    │   └── index.css
    ├── public/index.html
    ├── package.json
    └── README.md
```

## Configuration knobs

In `backend/app.py`:

- `OllamaLLM(model="llama3.2", temperature=0, num_ctx=8192, num_predict=512, keep_alive=-1)`

In `backend/vector.py`:

- `OllamaEmbeddings(model="mxbai-embed-large", keep_alive=-1)`
- `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
- Retriever: similarity search with `k=5`

## Troubleshooting

- **`ConnectionError` to `127.0.0.1:11434`** — Ollama isn't running. Start it (`ollama serve` or launch the app) and confirm `ollama list` works.
- **Slow first answer** — usually means the warmup didn't complete (Ollama wasn't up yet). Restart the backend after Ollama is running.
- **Empty answers** — check `/query`'s debug payload (`retrieved_count`); if zero, the PDF likely didn't text-extract well (scanned PDFs need OCR upstream).
- **CORS errors in the browser** — the backend uses `allow_origins=["*"]` for development; restrict before deploying.
- **Re-embed a PDF from scratch** — delete `backend/uploaded_files/<hash>_chroma/` and re-upload.

## License

MIT (or whatever you choose) — local-only educational project.
