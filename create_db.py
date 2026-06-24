import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_or_create_collection(name="documents")



results = collection.query(query_texts=["What is a Transformer?"], n_results=1)
print("Query results:", results)
#print(f"\nCollection has {collection.count()} documents.")
