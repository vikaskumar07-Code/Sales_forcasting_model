import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

df = pd.read_csv("notebook/data/Walmart.csv")


df["text"]=df.astype(str).agg("|".join,axis=1)

loader=DataFrameLoader(df,page_content_column="text")
documents=loader.load()

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embeddings ready")

vectorstore = FAISS.from_documents(documents, embeddings)

# save locally
vectorstore.save_local("faiss_index")

print("FAISS vector DB created & saved ")