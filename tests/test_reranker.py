from loaders.document_loader import load_document

from chunking.recursive_chunking import (
    recursive_chunk_documents
)

from reranking.reranker import (
    rerank_documents
)

documents = load_document(
    "data/document.pdf"
)

chunks = recursive_chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=100
)

candidate_chunks = chunks[:10]

question = "What is the main objective of the document?"

results = rerank_documents(
    question,
    candidate_chunks,
    top_n=3
)

for index, result in enumerate(results):

    document = result["document"]
    score = result["score"]

    print(f"\nRank {index + 1}")

    print("Score:")
    print(score)

    print("\nMetadata:")
    print(document.metadata)

    print("\nContent:")
    print(document.page_content[:500])