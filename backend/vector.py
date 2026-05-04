"""Utility helpers for building and querying a Chroma vector store from PDFs.

This module exposes functions so the backend can create a vector DB for an
uploaded PDF and then query it. The original script built a single DB at
import time; for a web service we need functions that create/reuse stores per
uploaded document.
"""

import os
import hashlib
import logging
from typing import List, Tuple
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger("pdf-agent.vector")

# Default embedding model used in the project
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"

# keep_alive=-1 keeps the embedding model resident in Ollama indefinitely so
# subsequent embed calls don't pay reload cost.
_embeddings = OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL, keep_alive=-1)


def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def create_retriever_from_pdf(
    pdf_path: str, persist_dir: str = None, collection_name: str = None
) -> Tuple[object, dict]:
    """Create (or load) a Chroma vector store from a PDF and return a Retriever.

    - pdf_path: path to the uploaded PDF file
    - persist_dir: directory where Chroma will persist DB files. If None,
      a content-hash-keyed folder is used so re-uploads of the same PDF
      reuse existing embeddings.
    - collection_name: optional Chroma collection name.

    Returns (retriever, info).
    """
    pdf_path = str(pdf_path)
    pdf_file = Path(pdf_path)
    file_hash = compute_file_hash(pdf_path)

    if persist_dir is None:
        persist_dir = str(pdf_file.parent / f"{file_hash}_chroma")

    os.makedirs(persist_dir, exist_ok=True)

    # Open (or create) the vector store first so we can check whether it
    # already has embeddings before re-parsing the PDF.
    vector_store = Chroma(
        collection_name=collection_name or "pdf_knowledge_base",
        persist_directory=persist_dir,
        embedding_function=_embeddings,
    )

    existing_count = 0
    try:
        existing_count = vector_store._collection.count()
    except Exception:
        try:
            existing_count = len(vector_store.get(limit=1).get("ids", []))
        except Exception:
            existing_count = 0

    documents = []
    if existing_count == 0:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logger.info("Loaded PDF %s with %d pages", pdf_file.name, len(pages))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = splitter.split_documents(pages)
        logger.info("Split PDF into %d document chunks", len(documents))

        logger.info("Creating new vector DB at %s from %s", persist_dir, pdf_file.name)
        vector_store.add_documents(documents)
        logger.info("Vector DB created successfully at %s", persist_dir)
    else:
        logger.info(
            "Reusing existing vector DB at %s (%d embeddings)", persist_dir, existing_count
        )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Lightweight debug payload. When we reused an existing store we don't
    # have the freshly-split docs; pull a few samples from Chroma instead so
    # the keyword-fallback path in app.py keeps working across restarts.
    samples: List[str] = []
    chunks: List[str] = []
    try:
        if documents:
            for d in documents[:3]:
                samples.append(getattr(d, "page_content", str(d)).replace("\n", " ")[:400])
            for d in documents:
                chunks.append(" ".join(getattr(d, "page_content", str(d)).split()))
        else:
            stored = vector_store.get()
            stored_docs = stored.get("documents") or []
            for txt in stored_docs[:3]:
                samples.append(str(txt).replace("\n", " ")[:400])
            for txt in stored_docs:
                chunks.append(" ".join(str(txt).split()))
    except Exception:
        logger.debug("Could not assemble document samples for debug")

    chunk_count = len(documents) if documents else existing_count
    info = {
        "chunk_count": chunk_count,
        "samples": samples,
        "chunks": chunks,
        "file_hash": file_hash,
        "persist_dir": persist_dir,
    }
    return retriever, info


def retrieve_documents(retriever, query: str, k: int = 5) -> List:
    """Return the list of relevant Document objects for a query using the retriever."""
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query)
        return docs[:k]

    # Older retriever API
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(query)
        return docs[:k]

    if hasattr(retriever, "retrieve"):
        docs = retriever.retrieve(query)
        return docs[:k]

    raise AttributeError("Retriever does not support document retrieval methods")
