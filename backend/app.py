import asyncio
import hashlib
import logging
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import create_retriever_from_pdf, retrieve_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pdf-agent")

# In-memory stores (single-process service)
DOCS = {}      # doc_id -> {"persist_dir", "filename", "retriever", "info", "file_hash"}
HASH_INDEX = {}  # file_hash -> doc_id  (lets us dedupe re-uploads of the same PDF)
SESSIONS = {}  # session_id -> {"doc_id", "history"}

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


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    session_id: Optional[str] = None
    doc_id: Optional[str] = None


def _resolve_retriever(req: QueryRequest):
    """Returns (retriever, session_or_none, doc_id)."""
    if req.session_id:
        session = SESSIONS.get(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="session_id not found")
        doc_id = session["doc_id"]
        return DOCS[doc_id]["retriever"], session, doc_id
    if req.doc_id:
        if req.doc_id not in DOCS:
            raise HTTPException(status_code=404, detail="doc_id not found; upload first")
        return DOCS[req.doc_id]["retriever"], None, req.doc_id
    raise HTTPException(status_code=400, detail="Provide session_id or doc_id")


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
    }
    HASH_INDEX[file_hash] = doc_id

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "took_seconds": took,
        "chunk_count": info.get("chunk_count"),
        "reused": False,
    }


@app.post("/start_session")
async def start_session(doc_id: str):
    logger.info("Start session requested for doc_id=%s", doc_id)
    if doc_id not in DOCS:
        logger.warning("start_session: doc_id not found: %s", doc_id)
        raise HTTPException(status_code=404, detail="doc_id not found; upload first")

    session_id = uuid4().hex
    SESSIONS[session_id] = {"doc_id": doc_id, "history": []}
    logger.info("Session started: %s -> doc %s", session_id, doc_id)
    return {"session_id": session_id}


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


@app.post("/query")
async def query(req: QueryRequest):
    retriever, session, doc_id = _resolve_retriever(req)
    logger.info(
        "Query request: session_id=%s doc_id=%s question=%s",
        req.session_id, req.doc_id, req.question,
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

    history_text = format_history(session["history"]) if session is not None else ""
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

    if session is not None:
        session["history"].append({"user": req.question, "assistant": answer})

    return {
        "answer": answer,
        "doc_id": doc_id,
        "debug": _gather_debug(docs, snippets, took_retrieval, took_model, doc_info),
    }


def _sse(event: str, data: str) -> str:
    # SSE: each line of payload must be prefixed with "data: ". Two newlines end the event.
    payload = "\n".join(f"data: {line}" for line in data.split("\n"))
    return f"event: {event}\n{payload}\n\n"


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    retriever, session, doc_id = _resolve_retriever(req)
    logger.info(
        "Stream query: session_id=%s doc_id=%s question=%s",
        req.session_id, req.doc_id, req.question,
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

    history_text = format_history(session["history"]) if session is not None else ""
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
        if session is not None:
            session["history"].append({"user": req.question, "assistant": answer})

        debug = _gather_debug(docs, snippets, took_retrieval, took_model, doc_info)
        # Final event carries the debug envelope so the client gets parity with /query.
        import json
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
        "sessions": list(SESSIONS.keys()),
    }
