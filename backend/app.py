import asyncio
import hashlib
import json
import logging
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

import storage
from vector import create_retriever_from_pdf, retrieve_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pdf-agent")

# In-memory caches of objects that can't live in SQLite (retrievers, the
# chunked text used by the keyword fallback). Hydrated from disk at startup
# and updated on /upload. Persistent state of record is the SQLite DB.
DOCS = {}        # doc_id -> {"persist_dir", "filename", "retriever", "info", "file_hash", "file_path"}
HASH_INDEX = {}  # file_hash -> doc_id

# OllamaLLM tuning:
#  - temperature=0       deterministic answers for RAG
#  - num_ctx=8192        room for 5x1000-char context chunks + history + question
#  - num_predict=512     cap output length so streaming doesn't run forever
#  - keep_alive=-1       keep model resident in Ollama (no reload between calls)
model = OllamaLLM(
    model="llama3.2",
    temperature=0,
    num_ctx=8192,
    num_predict=512,
    keep_alive=-1,
)

template = """You are an expert assistant that answers questions based on the provided PDF content.

Here is the relevant information from the PDF:
{context}

{history}Question: {question}

Answer clearly and concisely using only the provided context."""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def format_history(history, max_turns: int = 5) -> str:
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = ["Previous conversation:"]
    for turn in recent:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines) + "\n\n"


def _backfill_documents_from_disk() -> None:
    """Register PDFs that exist under uploaded_files/ but aren't in the DB yet.

    Filenames follow the {hash}_{original}.pdf convention written by /upload,
    paired with a {hash}_chroma/ persist dir. We only adopt entries that match
    that convention so we don't mistakenly index stray files.
    """
    uploads_dir = Path("uploaded_files")
    if not uploads_dir.exists():
        return
    for pdf in uploads_dir.glob("*.pdf"):
        name = pdf.name
        underscore = name.find("_")
        if underscore <= 0:
            continue
        file_hash = name[:underscore]
        if len(file_hash) != 16:
            continue
        persist_dir = uploads_dir / f"{file_hash}_chroma"
        if not persist_dir.exists():
            continue
        if storage.get_document_by_hash(file_hash):
            continue
        storage.upsert_document(
            doc_id=uuid4().hex,
            file_hash=file_hash,
            filename=name[underscore + 1:],
            file_path=str(pdf),
            persist_dir=str(persist_dir),
            chunk_count=None,
            size_bytes=pdf.stat().st_size,
        )
        logger.info("Backfilled existing PDF into storage: %s", name)


def _rehydrate_documents() -> None:
    """Rebuild DOCS/HASH_INDEX from persisted document rows.

    create_retriever_from_pdf is idempotent: when the persist_dir already has
    embeddings it reuses them without re-parsing. So we just call it again
    against each known PDF + persist_dir.
    """
    for row in storage.list_documents():
        file_path = row["file_path"]
        persist_dir = row["persist_dir"]
        if not Path(file_path).exists() or not Path(persist_dir).exists():
            logger.warning("Skipping rehydrate for %s: missing %s or %s",
                           row["filename"], file_path, persist_dir)
            continue
        try:
            retriever, info = create_retriever_from_pdf(file_path, persist_dir)
        except Exception as e:
            logger.warning("Failed to rehydrate %s: %s", row["filename"], e)
            continue
        DOCS[row["doc_id"]] = {
            "persist_dir": persist_dir,
            "filename": row["filename"],
            "retriever": retriever,
            "info": info,
            "file_hash": row["file_hash"],
            "file_path": file_path,
        }
        HASH_INDEX[row["file_hash"]] = row["doc_id"]
    if DOCS:
        logger.info("Rehydrated %d document(s) from storage", len(DOCS))


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init()
    try:
        await asyncio.to_thread(_backfill_documents_from_disk)
    except Exception as e:
        logger.warning("Document backfill failed: %s", e)
    try:
        await asyncio.to_thread(_rehydrate_documents)
    except Exception as e:
        logger.warning("Document rehydration failed: %s", e)

    # Best-effort warmup so the first user query doesn't pay the model load cost.
    try:
        await chain.ainvoke({"context": "ready", "question": "say ok", "history": ""})
        logger.info("Model warmup complete")
    except Exception as e:
        logger.warning("Model warmup skipped: %s", e)
    yield


app = FastAPI(title="PDF QA Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None
    doc_id: Optional[str] = None


class CreateChatRequest(BaseModel):
    doc_id: str
    title: Optional[str] = None


class RenameChatRequest(BaseModel):
    title: str


def _resolve_retriever(req: QueryRequest):
    """Returns (retriever, chat_or_none, doc_id)."""
    if req.chat_id:
        chat = storage.get_chat(req.chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat_id not found")
        doc_id = chat["doc_id"]
        if doc_id not in DOCS:
            raise HTTPException(status_code=404, detail="associated doc not loaded")
        return DOCS[doc_id]["retriever"], chat, doc_id
    if req.doc_id:
        if req.doc_id not in DOCS:
            raise HTTPException(status_code=404, detail="doc_id not found; upload first")
        return DOCS[req.doc_id]["retriever"], None, req.doc_id
    raise HTTPException(status_code=400, detail="Provide chat_id or doc_id")


def _build_context(retriever, question: str):
    """Sync helper run via to_thread; returns (docs, snippets, context_text, retrieval_time)."""
    start = time.time()
    docs = retrieve_documents(retriever, question, k=5)
    took = time.time() - start
    snippets = [getattr(d, "page_content", str(d))[:300].replace("\n", " ") for d in docs]
    context = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
    return docs, snippets, context, took


def _keyword_fallback(question: str, doc_info: dict) -> Optional[str]:
    if not doc_info or not doc_info.get("chunks"):
        return None
    qwords = [w for w in question.lower().split() if len(w) > 2]
    scored = []
    for chunk in doc_info["chunks"]:
        low = chunk.lower()
        score = sum(1 for w in qwords if w in low)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:3]]
    return "\n\n".join(top) if top else None


def _auto_title_from_question(question: str, max_len: int = 60) -> str:
    q = " ".join(question.split())
    if len(q) <= max_len:
        return q
    return q[: max_len - 1].rstrip() + "…"


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and create a vector store for it. Returns a doc_id to query."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    file_hash = hashlib.sha256(contents).hexdigest()[:16]

    # Dedup: if we've already indexed this exact PDF, return the existing doc_id.
    if file_hash in HASH_INDEX:
        existing_id = HASH_INDEX[file_hash]
        existing = DOCS[existing_id]
        logger.info("Re-using existing doc %s for hash %s", existing_id, file_hash)
        return {
            "doc_id": existing_id,
            "filename": existing["filename"],
            "took_seconds": 0.0,
            "chunk_count": existing["info"].get("chunk_count"),
            "reused": True,
        }

    uploads_dir = Path("uploaded_files")
    uploads_dir.mkdir(exist_ok=True)

    # Prefix saved filename with the hash so different PDFs sharing a name
    # don't clobber each other on disk.
    safe_name = f"{file_hash}_{Path(file.filename).name}"
    dest_path = uploads_dir / safe_name
    with dest_path.open("wb") as f:
        f.write(contents)

    persist_dir = str(uploads_dir / f"{file_hash}_chroma")

    logger.info("Upload received: %s -> %s (persist=%s)", file.filename, dest_path, persist_dir)
    start = time.time()
    try:
        retriever, info = await asyncio.to_thread(
            create_retriever_from_pdf, str(dest_path), persist_dir
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error creating retriever: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=str(e))
    took = time.time() - start
    logger.info("Retriever ready for %s (took %.2fs)", file.filename, took)

    doc_id = uuid4().hex
    DOCS[doc_id] = {
        "persist_dir": persist_dir,
        "filename": file.filename,
        "retriever": retriever,
        "info": info,
        "file_hash": file_hash,
        "file_path": str(dest_path),
    }
    HASH_INDEX[file_hash] = doc_id

    storage.upsert_document(
        doc_id=doc_id,
        file_hash=file_hash,
        filename=file.filename,
        file_path=str(dest_path),
        persist_dir=persist_dir,
        chunk_count=info.get("chunk_count"),
        size_bytes=len(contents),
    )

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "took_seconds": took,
        "chunk_count": info.get("chunk_count"),
        "reused": False,
    }


# ---------- documents ----------

@app.get("/documents")
def list_documents():
    docs_rows = storage.list_documents()
    return {"documents": [
        {
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "chunk_count": r["chunk_count"],
            "size_bytes": r["size_bytes"],
            "created_at": r["created_at"],
            "loaded": r["doc_id"] in DOCS,
        }
        for r in docs_rows
    ]}


@app.get("/pdf/{doc_id}")
def get_pdf(doc_id: str):
    row = storage.get_document(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="doc_id not found")
    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=410, detail="PDF file missing on disk")
    return FileResponse(path, media_type="application/pdf", filename=row["filename"])


# ---------- chats ----------

@app.get("/chats")
def list_chats():
    return {"chats": storage.list_chats()}


@app.post("/chats")
def create_chat(req: CreateChatRequest):
    if req.doc_id not in DOCS:
        # Allow creating chats for docs that exist in storage but aren't loaded
        # (e.g. PDF file deleted). Soft-check so the caller gets a clear error.
        if not storage.get_document(req.doc_id):
            raise HTTPException(status_code=404, detail="doc_id not found")
    chat = storage.create_chat(req.doc_id, req.title)
    return chat


@app.get("/chats/{chat_id}")
def get_chat(chat_id: str):
    chat = storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="chat_id not found")
    doc = storage.get_document(chat["doc_id"])
    return {
        "chat": chat,
        "document": doc,
        "messages": storage.list_messages(chat_id),
    }


@app.patch("/chats/{chat_id}")
def rename_chat(chat_id: str, req: RenameChatRequest):
    if not storage.rename_chat(chat_id, req.title):
        raise HTTPException(status_code=404, detail="chat_id not found")
    return storage.get_chat(chat_id)


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    if not storage.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="chat_id not found")
    return {"ok": True}


# ---------- query ----------

def _gather_debug(docs, snippets, took_retrieval, took_model, doc_info):
    debug_payload = {
        "retrieved_count": len(docs),
        "snippets": snippets,
        "retrieval_time_s": took_retrieval,
        "model_time_s": took_model,
    }
    if doc_info:
        debug_payload["corpus_chunk_count"] = doc_info.get("chunk_count")
        debug_payload["corpus_samples"] = doc_info.get("samples")
    return debug_payload


def _persist_turn(chat: Optional[dict], question: str, answer: str,
                  metrics: dict, sources: list) -> None:
    if not chat:
        return
    storage.add_message(chat_id=chat["chat_id"], role="user", content=question)
    storage.add_message(
        chat_id=chat["chat_id"], role="assistant", content=answer,
        metrics=metrics, sources=sources,
    )
    # Auto-name the chat from its first user message if not yet titled.
    if not chat.get("title"):
        storage.rename_chat(chat["chat_id"], _auto_title_from_question(question))


@app.post("/query")
async def query(req: QueryRequest):
    retriever, chat, doc_id = _resolve_retriever(req)
    logger.info(
        "Query request: chat_id=%s doc_id=%s question=%s",
        req.chat_id, req.doc_id, req.question,
    )

    try:
        docs, snippets, context, took_retrieval = await asyncio.to_thread(
            _build_context, retriever, req.question
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error during retrieval: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "Retrieved %d docs (retrieval took %.3fs). Context length=%d",
        len(docs), took_retrieval, len(context),
    )

    doc_info = DOCS.get(doc_id, {}).get("info") if doc_id else None

    if len(docs) == 0:
        logger.warning("No documents retrieved for '%s'. Trying keyword fallback", req.question)
        fb = _keyword_fallback(req.question, doc_info)
        if fb:
            logger.info("Fallback selected chunks for query")
            context = fb
        else:
            logger.warning("No fallback context available")

    history = storage.history_for_prompt(chat["chat_id"]) if chat else []
    history_text = format_history(history)
    inputs = {"context": context, "question": req.question, "history": history_text}

    try:
        start = time.time()
        result = await chain.ainvoke(inputs)
        took_model = time.time() - start
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Model invocation error: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=str(e))

    answer = result
    logger.info("Model returned (took %.3fs) — answer length=%d", took_model, len(str(answer)))

    debug = _gather_debug(docs, snippets, took_retrieval, took_model, doc_info)
    _persist_turn(chat, req.question, str(answer),
                  metrics={"retrieval_time_s": took_retrieval, "model_time_s": took_model},
                  sources=snippets)

    return {"answer": answer, "doc_id": doc_id, "debug": debug}


def _sse(event: str, data: str) -> str:
    # SSE: each line of payload must be prefixed with "data: ". Two newlines end the event.
    payload = "\n".join(f"data: {line}" for line in data.split("\n"))
    return f"event: {event}\n{payload}\n\n"


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    retriever, chat, doc_id = _resolve_retriever(req)
    logger.info(
        "Stream query: chat_id=%s doc_id=%s question=%s",
        req.chat_id, req.doc_id, req.question,
    )

    try:
        docs, snippets, context, took_retrieval = await asyncio.to_thread(
            _build_context, retriever, req.question
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error during retrieval: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=str(e))

    doc_info = DOCS.get(doc_id, {}).get("info") if doc_id else None

    if len(docs) == 0:
        fb = _keyword_fallback(req.question, doc_info)
        if fb:
            context = fb

    history = storage.history_for_prompt(chat["chat_id"]) if chat else []
    history_text = format_history(history)
    inputs = {"context": context, "question": req.question, "history": history_text}

    async def event_stream():
        full = []
        start = time.time()
        try:
            async for chunk in chain.astream(inputs):
                # langchain may yield strings or messages; normalize to text.
                text = chunk if isinstance(chunk, str) else getattr(chunk, "content", str(chunk))
                if not text:
                    continue
                full.append(text)
                yield _sse("token", text)
        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield _sse("error", str(e))
            return

        took_model = time.time() - start
        answer = "".join(full)
        debug = _gather_debug(docs, snippets, took_retrieval, took_model, doc_info)

        try:
            _persist_turn(chat, req.question, answer,
                          metrics={"retrieval_time_s": took_retrieval, "model_time_s": took_model},
                          sources=snippets)
        except Exception as e:
            logger.error("Failed to persist chat turn: %s", e)

        yield _sse("done", json.dumps({"doc_id": doc_id, "debug": debug}))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/status")
def status():
    return {
        "docs": {k: {"filename": v["filename"]} for k, v in DOCS.items()},
        "chats": [c["chat_id"] for c in storage.list_chats()],
    }
