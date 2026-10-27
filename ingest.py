import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def main():
    # 1. Load documents from the data directory
    print("Loading PDFs from data folder...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} raw document pages.")

    # 2. Split documents into manageable chunks
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Generated {len(chunks)} text chunks.")

    # 3. Create local embeddings using Hugging Face
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Save chunks into ChromaDB vector store
    print("Saving embeddings to local ChromaDB...")
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    print(f"Vector database successfully created and saved to {CHROMA_PATH}!")

if __name__ == "__main__":
    main()

