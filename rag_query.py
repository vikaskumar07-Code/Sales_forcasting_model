from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

LLM = OllamaLLM(model="phi3")

query = input("Ask your question: ")

docs = retriever.invoke(query)
context = "\n".join([doc.page_content for doc in docs])

prompt=f"""
You are a helpful AI assistant.

If the question is greeting, respond normally.
If the question is dataset related, use the context.
Only answer using the provided context.
If the answer is not clearly present, say "I don't know".


Context:
{context}

Question: {query}
"""


response = LLM.invoke(prompt)

print("\n AI Answer:\n")
print(response)


