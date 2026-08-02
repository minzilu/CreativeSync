import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
