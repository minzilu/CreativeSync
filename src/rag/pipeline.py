import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(data_dir="./corpus"):
    """
    Load text documents from the specified corpus directory.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()

def chunk_documents(documents):
    """
    Split loaded documents into manageable chunks for embeddings.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks, persist_directory="./chroma_db"):
    """
    Embed document chunks and persist vector database using Chroma.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def evaluate_retrieval(vectorstore, query):
    """
    Evaluate similarity search retrieval for a given user query.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)
    print(f"Query: {query}")
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:150]}...\n")
    return results

if __name__ == "__main__":
    docs = load_documents()
    if docs:
        chunks = chunk_documents(docs)
        db = create_vector_store(chunks)
        evaluate_retrieval(db, "What lighting gear is required for an outdoor sunset shoot?")
    else:
        print("No documents found in corpus directory. Add .txt files to ./corpus to build vector database.")
