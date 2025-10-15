from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import os

# ====== 1. Load PDF ======
# Resolve the PDF path relative to this file so imports work from any CWD
pdf_path = os.path.join(os.path.dirname(__file__), "Employee-Handbook.pdf")
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# ====== 2. Split PDF into chunks ======
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = splitter.split_documents(pages)

# ====== 3. Create embeddings ======
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ====== 4. Create / Load Vector DB ======
db_location = "./pdf_chroma_db"
db_exists = os.path.exists(db_location)

vector_store = Chroma(
    collection_name="pdf_knowledge_base",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Only add if DB not already built
if not db_exists:
    print("Creating new vector database from PDF...")
    vector_store.add_documents(documents)
    print("✅ Vector DB created successfully!")

# ====== 5. Create Retriever ======
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
