"""Utility helpers for building and querying a Chroma vector store from PDFs.

This module exposes functions so the backend can create a vector DB for an
uploaded PDF and then query it. The original script built a single DB at
import time; for a web service we need functions that create/reuse stores per
uploaded document.
"""

import os
import logging
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger("pdf-agent.vector")

# Default embedding model used in the project
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"


def create_retriever_from_pdf(pdf_path: str, persist_dir: str = None, collection_name: str = None):
    """Create (or load) a Chroma vector store from a PDF and return a Retriever.

    - pdf_path: path to the uploaded PDF file
    - persist_dir: directory where Chroma will persist DB files. If None,
      a folder next to the PDF will be used: `<pdf_parent>/<pdf_stem>_chroma`.
    - collection_name: optional Chroma collection name.

    Returns a langchain Retriever instance.
    """
    pdf_path = str(pdf_path)
    pdf_file = Path(pdf_path)

    if persist_dir is None:
        persist_dir = str(pdf_file.parent / f"{pdf_file.stem}_chroma")

    os.makedirs(persist_dir, exist_ok=True)

    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    logger.info("Loaded PDF %s with %d pages", pdf_file.name, len(pages))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(pages)
    logger.info("Split PDF into %d document chunks", len(documents))

    # Embeddings and vector store
    embeddings = OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=collection_name or "pdf_knowledge_base",
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # If the store is empty, add documents
    # Chroma persistence layout varies; a simple heuristic is to check for files
    # inside the persist_dir. If empty, add documents.
    dir_is_empty = not any(Path(persist_dir).iterdir())
    if dir_is_empty:
        logger.info("Creating new vector DB at %s from %s", persist_dir, pdf_file.name)
        vector_store.add_documents(documents)
        try:
            vector_store.persist()
        except Exception:
            # Some Chroma bindings persist automatically; ignore failures.
            logger.debug("Chroma persist() call failed or not supported; continuing")
        logger.info("Vector DB created successfully at %s", persist_dir)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Prepare small info payload for debugging: number of chunks and a few samples
    samples = []
    try:
        for d in documents[:3]:
            txt = getattr(d, "page_content", str(d))
            samples.append(txt.replace("\n", " ")[:400])
    except Exception:
        logger.debug("Could not create document samples for debug")

    # Keep a lightweight list of chunk texts for debugging and simple fallbacks.
    chunks = []
    try:
        for d in documents:
            txt = getattr(d, "page_content", str(d))
            # keep full chunk but trim excessive whitespace
            chunks.append(" ".join(txt.split()))
    except Exception:
        logger.debug("Could not serialize all document chunks for debug")

    info = {"chunk_count": len(documents), "samples": samples, "chunks": chunks}
    return retriever, info


def retrieve_documents(retriever, query: str, k: int = 5) -> List:
    """Return the list of relevant Document objects for a query using the retriever."""
    # LangChain retrievers typically implement get_relevant_documents
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(query)
        return docs[:k]

    # Some retrievers expose `retrieve`
    if hasattr(retriever, "retrieve"):
        docs = retriever.retrieve(query)
        return docs[:k]

    # Fallback: if retriever was created with a custom .invoke (older code), try that
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)

    raise AttributeError("Retriever does not support document retrieval methods")
