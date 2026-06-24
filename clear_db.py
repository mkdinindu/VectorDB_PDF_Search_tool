import sys
import argparse
import shutil
import chromadb
from chromadb.config import Settings


def main():
    parser = argparse.ArgumentParser(description="Clear the vector database")
    parser.add_argument("--collection", default="documents", help="Collection to clear (default: documents)")
    parser.add_argument("--hard", action="store_true", help="Delete the entire chroma_db directory")
    args = parser.parse_args()

    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )

    if args.hard:
        confirm = input("Delete entire chroma_db directory? This cannot be undone. (yes/no): ")
        if confirm.lower() == "yes":
            shutil.rmtree("./chroma_db")
            print("chroma_db directory deleted.")
        else:
            print("Cancelled.")
        return

    try:
        client.delete_collection(name=args.collection)
        print(f"Deleted collection '{args.collection}'.")
    except Exception:
        print(f"Collection '{args.collection}' does not exist.")


if __name__ == "__main__":
    main()
