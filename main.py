import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
client = chromadb.Client()

palm_embedding = embedding_functions.GooglePalmEmbeddingFunction(
    api_key=os.getenv("API_KEY"))

collection = client.create_collection(
    "all-my-documents", embedding_function=palm_embedding)

collection.add(
    documents=["This is document1", "This is document2"],
    metadatas=[{"source": "notion"}, {"source": "google-docs"}],
    ids=["doc1", "doc2"],
)

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,

    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

print(results)
