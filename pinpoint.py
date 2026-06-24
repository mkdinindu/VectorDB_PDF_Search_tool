import sys
import os
import argparse
import subprocess
import chromadb
from chromadb.config import Settings


def main():
    parser = argparse.ArgumentParser(description="Query and pinpoint locations in the source PDF")
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--collection", default="documents", help="Collection name (default: documents)")
    parser.add_argument("--n-results", type=int, default=3, help="Number of results (default: 3)")
    parser.add_argument("--open", action="store_true", help="Open PDF at the result page")
    args = parser.parse_args()

    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(name=args.collection)

    if collection.count() == 0:
        print("Collection is empty. Ingest some documents first.")
        sys.exit(1)

    results = collection.query(
        query_texts=[args.query],
        n_results=args.n_results,
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    print(f"Query: \"{args.query}\"\n")
    print(f"Top {len(documents)} results:\n")

    for i, (doc_id, doc_text, dist, meta) in enumerate(zip(ids, documents, distances, metadatas), 1):
        confidence = 1 - dist
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        filename = meta.get("filename", "?")

        print(f"{i}. [Score: {confidence:.4f}]")
        print(f"   Source: {source}")
        print(f"   Page: {page}")
        print(f"   Content: {doc_text[:300]}{'...' if len(doc_text) > 300 else ''}")
        print()

        if args.open and os.path.isfile(source):
            try:
                subprocess.Popen(["xdg-open", source])
                print(f"   >>> Opened {source} (page {page})")
            except FileNotFoundError:
                print("   >>> Could not open PDF (xdg-open not available)")
            print()


if __name__ == "__main__":
    main()
