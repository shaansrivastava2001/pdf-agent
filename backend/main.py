from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import create_retriever_from_pdf, retrieve_documents
import os

# Example CLI script using the refactored vector functions
# For a proper service, use: uvicorn app:app --reload --host 127.0.0.1 --port 8000

# Initialize model
model = OllamaLLM(model="llama3.2")

# Define prompt template
template = """
You are an expert assistant that answers questions based on the provided PDF content.

Here is the relevant information from the PDF:
{context}

Question: {question}

Answer clearly and concisely using only the provided context.
"""

prompt = ChatPromptTemplate.from_template(template)

# Combine into chain
chain = prompt | model

# Use Employee-Handbook.pdf if it exists, otherwise ask for path
pdf_path = os.path.join(os.path.dirname(__file__), "Employee-Handbook.pdf")
if not os.path.exists(pdf_path):
    pdf_path = input("Enter path to PDF file: ").strip()

print(f"Loading PDF: {pdf_path}")
retriever, _ = create_retriever_from_pdf(pdf_path)

# Interactive Q&A loop
while True:
    print("\n")
    question = input("Ask a question (or type 'q' to quit): ").strip()
    if question.lower() in ['q', 'exit']:
        print("Goodbye!")
        break

    # Retrieve relevant chunks from PDF
    retrieved_docs = retrieve_documents(retriever, question, k=5)
    context = "\n\n".join([getattr(doc, 'page_content', str(doc)) for doc in retrieved_docs])

    # Run the model
    result = chain.invoke({"context": context, "question": question})

    # Display result
    print("\n💬 Answer:")
    print(result)
