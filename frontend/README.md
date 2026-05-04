# Frontend for PDF Agent

A minimal React app (bundled with **Parcel**) for the PDF Agent backend. Streams answers from the backend's `/query/stream` SSE endpoint and renders them token-by-token.

## Prerequisites

- Node.js 18+ and npm.
- The backend running at `http://127.0.0.1:8000` (see `backend/README.md`).

## Install

From the project root:

```bash
cd frontend
npm install
```

## Run the dev server

```bash
npm start
```

Parcel serves the app on <http://localhost:5173> with hot reload. The dev server proxies nothing — the frontend talks to the backend directly via CORS, so make sure the backend is up first.

## Production build

```bash
npm run build
```

Output goes to `frontend/build/` (an `index.html` plus hashed JS/CSS bundles). Serve it with any static file server, e.g.:

```bash
npx serve build
```

## Configure the backend URL

By default the app posts to `http://127.0.0.1:8000`. To override, edit the meta tag in `frontend/public/index.html`:

```html
<meta name="api-base" content="http://your-backend:8000" />
```

The app reads that tag at startup and falls back to `http://127.0.0.1:8000` if it isn't present.

## What the UI does

1. **Upload PDF** → `POST /upload` (multipart). On a re-upload of the same file the backend returns `reused: true` and the UI shows "Reused existing index…".
2. **Auto start session on first question** → `POST /start_session?doc_id=…` (so chat history is preserved across turns).
3. **Ask** → `POST /query/stream` (SSE). Tokens are appended to the assistant message as they arrive; the message list auto-scrolls.

## Notes

- The backend has CORS enabled for development (`allow_origins=["*"]`); restrict it before deploying anywhere public.
- The first upload of a PDF takes time (PDF parsing + embedding). Identical re-uploads are effectively instant thanks to content-hash dedup on the backend.
- If a question hangs on "Thinking…" with no tokens, check that Ollama is running and the model (`llama3.2`) is pulled — see `backend/README.md`.
