import os
import hashlib

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings


DB_PATH = "./chroma_db"


def get_client():
    return chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection_count():
    try:
        return get_client().get_or_create_collection(name="documents").count()
    except Exception:
        return 0


def extract_pages_from_pdf(path: str) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text:
            pages.append((i, text))
    return pages


def chunk_pdf_pages(
    pages: list[tuple[int, str]], chunk_size: int, overlap: int, basename: str, source_path: str
) -> tuple[list[str], list[str], list[dict]]:
    ids, documents, metadatas = [], [], []
    chunk_idx = 0

    for page_num, page_text in pages:
        start = 0
        while start < len(page_text):
            end = start + chunk_size
            chunk = page_text[start:end]
            if chunk.strip():
                doc_id = hashlib.md5(f"{basename}_{chunk_idx}".encode()).hexdigest()[:12]
                ids.append(doc_id)
                documents.append(chunk)
                metadatas.append({
                    "source": source_path,
                    "page": page_num,
                    "char_start": start,
                    "char_end": end,
                    "chunk": chunk_idx,
                    "filename": basename,
                })
                chunk_idx += 1
            start += chunk_size - overlap

    return ids, documents, metadatas


def add_to_collection(documents, metadatas, ids, collection_name="documents"):
    client = get_client()
    client.get_or_create_collection(name=collection_name).add(
        documents=documents, metadatas=metadatas, ids=ids
    )


def query_collection(query: str, n_results: int = 3, collection_name="documents"):
    client = get_client()
    collection = client.get_collection(name=collection_name)
    if collection.count() == 0:
        return None
    results = collection.query(query_texts=[query], n_results=n_results)
    return results

def similar_terms(query: str, n_results: int = 10, collection_name="documents"):
    results = query_collection(query, n_results=n_results, collection_name=collection_name)
    if results is None:
        return None
    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]

    exact_matches = []
    alternative_matches = []

    term_lower = query.lower()

    for doc, dist, meta in zip(docs, dists, metas):
        entry = {"text": doc, "distance": dist, "metadata": meta}
        if term_lower in doc.lower():
            exact_matches.append(entry)
        else:
            alternative_matches.append(entry)
    return {
        "query": query,
        "exact_matches":exact_matches,
        "alternative_matches": alternative_matches,
    }

def get_all_embeddings(collection_name="documents"):
    client = get_client()
    collection = client.get_or_create_collection(name=collection_name)
    if collection.count() == 0:
        return None
    result = collection.get(include=["documents", "embeddings", "metadatas"])
    return result["ids"], result["documents"], result["embeddings"], result["metadatas"]


def clear_collection(collection_name="documents"):
    get_client().delete_collection(name=collection_name)
