# Frontend for PDF Agent

This is a minimal React + Vite frontend for the PDF Agent backend.

Quick start (from project root):

```bash
cd frontend
npm install
npm start
```

By default the frontend expects the backend at `http://127.0.0.1:8000`.

To change the backend URL you can either:

- Edit the meta tag in `frontend/public/index.html` (<meta name="api-base" content="http://your-backend:8000" />), or
- At runtime, set `window.__API_BASE__ = 'http://your-backend:8000'` before the app loads.

The app reads the meta tag at startup and falls back to `http://127.0.0.1:8000` if not present.

The UI lets you:
- Upload a PDF (POST /upload)
- Start a session (POST /start_session)
- Ask questions (POST /query)

Notes:
- The backend must allow CORS (the server in `backend/app.py` has CORS enabled for development).
- The upload process can take time while building embeddings; wait for the upload response before starting a session.
