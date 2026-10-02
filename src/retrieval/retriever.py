from vectorstore.chroma_manager import (
    get_vectorstore
)

from reranking.reranker import (
    DocumentReranker
)

RETRIEVAL_K = 10
RERANK_TOP_N = 3

reranker = DocumentReranker()

def retrieve_and_rerank(
    question,
    collection_name
):

    vectorstore = get_vectorstore(
        collection_name
    )

    candidates = vectorstore.similarity_search(
        question,
        k=RETRIEVAL_K
    )

    ranked_results = reranker.rerank(
        question,
        candidates,
        top_n=RERANK_TOP_N
    )

    return ranked_results