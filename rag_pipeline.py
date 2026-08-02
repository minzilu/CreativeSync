"""
Redirect root import to modular RAG pipeline package.
"""
from src.rag.pipeline import load_documents, chunk_documents, create_vector_store, evaluate_retrieval

if __name__ == "__main__":
    from src.rag.pipeline import main
    docs = load_documents()
    if docs:
        chunks = chunk_documents(docs)
        db = create_vector_store(chunks)
        evaluate_retrieval(db, "What lighting gear is required for an outdoor sunset shoot?")