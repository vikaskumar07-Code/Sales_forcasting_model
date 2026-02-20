import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading dataset...")

df = pd.read_csv("notebook/data/Walmart.csv")

df["text"] = df.apply(
    lambda row: f"""
    Walmart sales record:
    Store: {row['Store']}
    Date: {row['Date']}
    Weekly Sales: {row['Weekly_Sales']}
    Holiday: {row['Holiday_Flag']}
    Temperature: {row['Temperature']}
    Fuel Price: {row['Fuel_Price']}
    CPI: {row['CPI']}
    Unemployment: {row['Unemployment']}
    """,
    axis=1
)

loader = DataFrameLoader(
    df,
    page_content_column="text"
)

documents = loader.load()

#Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60
)

documents = splitter.split_documents(documents)

print("Documents created:", len(documents))

#Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings ready")

#FAISS
vectorstore = FAISS.from_documents(documents, embeddings)

vectorstore.save_local("faiss_index")

print("FAISS vector DB created & saved")