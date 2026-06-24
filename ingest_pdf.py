import sys
import os
import argparse
import hashlib

from pypdf import PdfReader

from chromadb.config import Settings
import chromadb


def extract_pages_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text:
            pages.append((i, text))
    return pages


def chunk_page_text(page_num: int, text: str, chunk_size: int = 500, overlap: int = 100) -> list[tuple[int, int, int, str]]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((page_num, start, end, chunk))
        start += chunk_size - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the Chroma vector database")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--chunk-size", type=int, default=500, help="Character chunk size (default: 500)")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap (default: 100)")
    parser.add_argument("--collection", default="documents", help="Collection name (default: documents)")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        print(f"Error: file not found: {args.pdf_path}")
        sys.exit(1)

    print(f"Extracting text from: {args.pdf_path}")
    pages = extract_pages_from_pdf(args.pdf_path)
    if not pages:
        print("Error: no text could be extracted from the PDF")
        sys.exit(1)
    print(f"Extracted {len(pages)} pages")

    basename = os.path.splitext(os.path.basename(args.pdf_path))[0]

    ids = []
    documents = []
    metadatas = []

    chunk_idx = 0
    for page_num, page_text in pages:
        page_chunks = chunk_page_text(page_num, page_text, args.chunk_size, args.overlap)
        for pn, char_start, char_end, chunk in page_chunks:
            doc_id = hashlib.md5(f"{basename}_{chunk_idx}".encode()).hexdigest()[:12]
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "source": args.pdf_path,
                "page": pn,
                "char_start": char_start,
                "char_end": char_end,
                "chunk": chunk_idx,
                "filename": basename,
            })
            chunk_idx += 1

    print(f"Split into {chunk_idx} chunks")

    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(name=args.collection)
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"Added {chunk_idx} chunks to collection '{args.collection}'")
    print(f"Collection now has {collection.count()} documents total")


if __name__ == "__main__":
    main()
