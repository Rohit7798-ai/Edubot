# create_memory_for_llm.py

import os
import logging
import warnings
from pathlib import Path

# Suppress noisy library warnings
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load Raw PDFs
BASE_DIR = Path(__file__).resolve().parent
# Use data/ subfolder if it exists, otherwise use the script's directory
DATA_PATH = str(BASE_DIR / "data") if (BASE_DIR / "data").is_dir() else str(BASE_DIR)

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


if __name__ == "__main__":
    print(f"Loading PDFs from: {DATA_PATH}...")
    documents = load_pdf_files(DATA_PATH)
    print("Length of PDF Pages:", len(documents))

    # Step 2: Split Documents into Chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    text_chunks = text_splitter.split_documents(documents)
    print("Number of Text Chunks:", len(text_chunks))

    # Step 3 : Create Vector Embeddings
    print("Loading embedding model...")
    def get_embedding_model():
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embedding_model 
    embedding_model = get_embedding_model()

    # Step 4 : store Embeddings in FAISS
    DB_FAISS_PATH = str(BASE_DIR / "vectorstore" / "db_faiss")
    print(f"Creating and saving FAISS vector store to: {DB_FAISS_PATH}...")
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("FAISS Vector Store Saved Successfully!!")
