# rag_query.py
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import re
import streamlit as st

# Load dataset
df = pd.read_csv("notebook/data/Walmart.csv")
df["Weekly_Sales"] = (
    df["Weekly_Sales"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# -----------------------------
# Cached objects to speed up
# -----------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

@st.cache_resource
def load_llm():
    return OllamaLLM(model="phi3")  # your LLM

vectorstore = load_vectorstore()
LLM = load_llm()

# -----------------------------
# Helper functions
# -----------------------------
def is_numeric_question(q):
    keywords = ["average","mean","sum","total","max","highest","lowest","minimum","maximum"]
    return any(k in q.lower() for k in keywords)

def extract_store(q):
    match = re.search(r"store\s*(\d+)", q.lower())
    if match:
        return int(match.group(1))
    return None

def numeric_engine(query):
    store = extract_store(query)
    data = df.copy()
    if store:
        data = data[data["Store"] == store]
    q = query.lower()
    if "average" in q or "mean" in q:
        return f"Average weekly sales: {data['Weekly_Sales'].mean():.2f}"
    if "total" in q or "sum" in q:
        return f"Total weekly sales: {data['Weekly_Sales'].sum():.2f}"
    if "highest" in q or "max" in q:
        row = data.loc[data['Weekly_Sales'].idxmax()]
        return f"Highest sales {row['Weekly_Sales']} on {row['Date']} for store {row['Store']}"
    if "lowest" in q or "min" in q:
        row = data.loc[data['Weekly_Sales'].idxmin()]
        return f"Lowest sales {row['Weekly_Sales']} on {row['Date']} for store {row['Store']}"
    return None

# -----------------------------
# Main RAG + numeric function
# -----------------------------
def ask_question(query: str):
    # -------------------
    # Numeric reasoning
    # -------------------
    if is_numeric_question(query):
        numeric_answer = numeric_engine(query)
        if numeric_answer:
            # Skip LLM explanation to save time
            return numeric_answer

    # -------------------
    # RAG retrieval
    # -------------------
    # Use FAISS directly (fast)
    docs = vectorstore.similarity_search(query, k=5)  # top 5 docs
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a Walmart sales analyst AI.

Rules:
- Use ONLY the context
- If greeting → respond normally
- If analytical question → infer patterns from context
- Give short, data-grounded answers
- If unsure → say I don't know

Context:
{context}

Question: {query}

Answer:
"""
    # Only one LLM call
    response = LLM.invoke(prompt)
    return response

# -----------------------------
# Standalone test
# -----------------------------
if __name__=="__main__":
    query = input("Ask your question: ")
    print("\nAI Answer:\n")
    print(ask_question(query))