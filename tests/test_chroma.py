from vectorstore.chroma_manager import get_vectorstore

collections = [
    "fixed_collection",
    "recursive_collection",
    "semantic_collection"
]

for collection_name in collections:

    print("\n========================================")
    print(collection_name)

    vectorstore = get_vectorstore(collection_name)
    collection = vectorstore._collection
    data = collection.get(
        include=[
            "documents",
            "embeddings",
            "metadatas"
        ]
    )

    print("Number of stored documents:", len(data["documents"]))

    if data["documents"]:

        print("\nFirst document:")
        print(data["documents"][0][:300])

        print("\nMetadata:")
        print(data["metadatas"][0])

        print("\nEmbedding:")
        print(data["embeddings"][0][:10])

        print("\nEmbedding dimensions:")
        print(len(data["embeddings"][0]))