from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # retriever you built from PDF

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

# Interactive Q&A loop
while True:
    print("\n")
    question = input("Ask a question (or type 'q' to quit): ").strip()
    if question.lower() in ['q', 'exit']:
        print("Goodbye!")
        break

    # Retrieve relevant chunks from PDF
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Run the model
    result = chain.invoke({"context": context, "question": question})

    # Display result
    print("\n💬 Answer:")
    print(result)
