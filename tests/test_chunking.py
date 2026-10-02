from loaders.document_loader import load_document
from chunking.fixed_chunking import fixed_chunk_documents
from chunking.recursive_chunking import recursive_chunk_documents


documents = load_document("data/document.pdf")

print(f"Number of documents/pages: {len(documents)}")

fixed_chunks = fixed_chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=100
)

print(f"Number of chunks: {len(fixed_chunks)}")

for i, chunk in enumerate(fixed_chunks[:3]):

    print("\n------------------------------")
    print(f"Fixed Chunk {i + 1}")

    print("Metadata:")
    print(chunk.metadata)

    print("\nContent:")
    print(chunk.page_content[:500])

recursive_chunks = recursive_chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=100
)

print(f"Number of chunks: {len(recursive_chunks)}")

for i, chunk in enumerate(recursive_chunks[:3]):

    print("\n------------------------------")
    print(f"Recursive Chunk {i + 1}")

    print("Metadata:")
    print(chunk.metadata)

    print("\nContent:")
    print(chunk.page_content[:500])