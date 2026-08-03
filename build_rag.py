from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def build_vector_store():
    print("Loading documents...")
    loader = TextLoader("data/knowledge_base.txt")
    documents = loader.load()

    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks. Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("data/faiss_index")
    print("FAISS vector store saved successfully!")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    build_vector_store()