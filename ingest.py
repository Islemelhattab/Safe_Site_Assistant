import hashlib
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("./data")
PERSIST_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def compute_doc_hash(doc: Document) -> str:
    """Generate a deterministic MD5 hash based on content and source metadata."""
    hasher = hashlib.md5()
    hasher.update(doc.page_content.encode("utf-8"))
    for k in sorted(doc.metadata.keys()):
        hasher.update(f"{k}:{doc.metadata[k]}".encode("utf-8"))
    return hasher.hexdigest()


def load_documents() -> List[Document]:
    """Load Markdown and TXT files from data directory."""
    raw_docs = []
    for file_path in DATA_DIR.glob("**/*"):
        if file_path.suffix.lower() in [".md", ".txt"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["file"] = file_path.name
                doc.metadata["page"] = doc.metadata.get("page", 1)
            raw_docs.extend(loaded)
    return raw_docs


def ingest_data():
    """
    Chunking Justification:
    - chunk_size = 500: Safety standards and procedures contain granular, standalone
      clauses (e.g., clearance heights, PPE specs). A 500-character size isolates
      individual instructions without diluting the embedding space.
    - chunk_overlap = 100: Retains continuous contextual phrasing across boundaries,
      preventing sentence or rule severance.
    """
    raw_docs = load_documents()
    if not raw_docs:
        print("[!] No documents found in data/. Aborting.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n## ", "\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # Prevent chunk duplication
    existing_records = vectordb.get(include=["metadatas"])
    existing_ids = set(existing_records.get("ids", []))

    new_chunks = []
    new_ids = []
    for chunk in chunks:
        chunk_id = compute_doc_hash(chunk)
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        vectordb.add_documents(documents=new_chunks, ids=new_ids)
        print(f"[+] Successfully ingested {len(new_chunks)} new chunks.")
    else:
        print("[✓] Vector store is up-to-date. No new chunks added.")


if __name__ == "__main__":
    ingest_data()