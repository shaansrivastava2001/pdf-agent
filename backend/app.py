from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4
from pathlib import Path
import shutil

from vector import create_retriever_from_pdf, retrieve_documents
import logging
import traceback
import time

# Configure simple logging for debug
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pdf-agent")

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="PDF QA Service")

# Allow CORS so the frontend can call this API during development
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores (for a simple, single-process service)
DOCS = {}  # doc_id -> {"persist_dir": str, "filename": str, "retriever": retriever}
SESSIONS = {}  # session_id -> {"doc_id": str, "history": list}

# Initialize model and prompt template once
model = OllamaLLM(model="llama3.2")

template = """
You are an expert assistant that answers questions based on the provided PDF content.

Here is the relevant information from the PDF:
{context}

Question: {question}

Answer clearly and concisely using only the provided context.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    doc_id: Optional[str] = None


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and create a vector store for it. Returns a doc_id to query."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    uploads_dir = Path("uploaded_files")
    uploads_dir.mkdir(exist_ok=True)

    dest_path = uploads_dir / file.filename
    # Save uploaded file
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Create a persist directory next to the file
    persist_dir = str(uploads_dir / f"{dest_path.stem}_chroma")

    logger.info("Upload received: %s -> %s (persist=%s)", file.filename, dest_path, persist_dir)
    # Build or load retriever (may take time)
    start = time.time()
    try:
        retriever, info = create_retriever_from_pdf(str(dest_path), persist_dir=persist_dir)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error creating retriever: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=str(e))
    took = time.time() - start
    logger.info("Retriever ready for %s (took %.2fs)", file.filename, took)

    doc_id = uuid4().hex
    DOCS[doc_id] = {"persist_dir": persist_dir, "filename": file.filename, "retriever": retriever, "info": info}

    return {"doc_id": doc_id, "filename": file.filename, "took_seconds": took, "chunk_count": info.get("chunk_count")}


@app.post("/start_session")
def start_session(doc_id: str):
    logger.info("Start session requested for doc_id=%s", doc_id)
    if doc_id not in DOCS:
        logger.warning("start_session: doc_id not found: %s", doc_id)
        raise HTTPException(status_code=404, detail="doc_id not found; upload first")

    session_id = uuid4().hex
    SESSIONS[session_id] = {"doc_id": doc_id, "history": []}
    logger.info("Session started: %s -> doc %s", session_id, doc_id)
    return {"session_id": session_id}


@app.post("/query")
def query(req: QueryRequest):
    # Resolve retriever and session
    retriever = None
    session = None

    if req.session_id:
        session = SESSIONS.get(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="session_id not found")
        doc_id = session["doc_id"]
        retriever = DOCS[doc_id]["retriever"]
    elif req.doc_id:
        if req.doc_id not in DOCS:
            raise HTTPException(status_code=404, detail="doc_id not found; upload first")
        retriever = DOCS[req.doc_id]["retriever"]
    else:
        raise HTTPException(status_code=400, detail="Provide session_id or doc_id")

    logger.info("Query request: session_id=%s doc_id=%s question=%s", req.session_id, req.doc_id, req.question)

    # Retrieve context
    try:
        start = time.time()
        docs = retrieve_documents(retriever, req.question, k=5)
        took_retrieval = time.time() - start
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error during retrieval: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=str(e))

    snippets = []
    for d in docs:
        txt = getattr(d, "page_content", str(d))
        snippets.append(txt[:300].replace("\n", " "))

    context = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
    logger.info("Retrieved %d docs (retrieval took %.3fs). Context length=%d", len(docs), took_retrieval, len(context))

    # If retrieval returned zero documents, include corpus info to help debug
    doc_info = None
    if session is not None:
        doc_id = session["doc_id"]
        doc_info = DOCS.get(doc_id, {}).get("info")
    else:
        if req.doc_id:
            doc_info = DOCS.get(req.doc_id, {}).get("info")

    if len(docs) == 0:
        logger.warning("No documents retrieved for query '%s'. Attempting keyword fallback", req.question)
        # If retriever returned nothing, try a simple keyword overlap fallback
        fallback_context = None
        if doc_info and doc_info.get("chunks"):
            qwords = [w for w in req.question.lower().split() if len(w) > 2]
            scores = []
            for i, chunk in enumerate(doc_info.get("chunks", [])):
                low = chunk.lower()
                score = 0
                for w in qwords:
                    if w in low:
                        score += 1
                if score > 0:
                    scores.append((score, i, chunk))
            scores.sort(reverse=True)
            top = [c for _, _, c in scores[:3]]
            if top:
                fallback_context = "\n\n".join(top)
                logger.info("Fallback selected %d chunks for query", len(top))
        if fallback_context:
            # Use fallback context for model invocation
            context = fallback_context
        else:
            logger.warning("No fallback context available for query '%s'", req.question)

    # Build input for chain
    inputs = {"context": context, "question": req.question}

    try:
        start = time.time()
        result = chain.invoke(inputs)
        took_model = time.time() - start
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Model invocation error: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=str(e))

    answer = result
    logger.info("Model returned (took %.3fs) — answer length=%d", took_model, len(str(answer)))

    # Save to history if session exists
    if session is not None:
        session["history"].append({"user": req.question, "assistant": answer})

    debug_payload = {"retrieved_count": len(docs), "snippets": snippets, "retrieval_time_s": took_retrieval, "model_time_s": took_model}
    if doc_info:
        debug_payload["corpus_chunk_count"] = doc_info.get("chunk_count")
        debug_payload["corpus_samples"] = doc_info.get("samples")

    # Return answer and debug info to frontend
    return {"answer": answer, "doc_id": session["doc_id"] if session else req.doc_id, "debug": debug_payload}


@app.get("/status")
def status():
    return {"docs": {k: {"filename": v["filename"]} for k, v in DOCS.items()}, "sessions": list(SESSIONS.keys())}
