from sklearn.metrics.pairwise import cosine_similarity

RELEVANCE_THRESHOLD = 0.50

def calculate_chunk_similarity(
    reference_answer,
    document,
    embedding_model
):
    reference_embedding = embedding_model.embed_query(
        reference_answer
    )

    chunk_embedding = embedding_model.embed_query(
        document.page_content
    )

    score = cosine_similarity(
        [reference_embedding],
        [chunk_embedding]
    )[0][0]

    return float(score)


def calculate_recall_at_k(
    reference_answer,
    retrieved_documents,
    embedding_model,
    k
):
    if not retrieved_documents:
        return 0.0

    top_k_documents = retrieved_documents[:k]

    for document in top_k_documents:

        similarity = calculate_chunk_similarity(
            reference_answer,
            document,
            embedding_model
        )

        if similarity >= RELEVANCE_THRESHOLD:
            return 1.0

    return 0.0


def calculate_mrr(
    reference_answer,
    retrieved_documents,
    embedding_model
):
    if not retrieved_documents:
        return 0.0

    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):

        similarity = calculate_chunk_similarity(
            reference_answer,
            document,
            embedding_model
        )

        if similarity >= RELEVANCE_THRESHOLD:

            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    reference_answer,
    retrieved_documents,
    embedding_model,
    k=10
):
    recall = calculate_recall_at_k(
        reference_answer,
        retrieved_documents,
        embedding_model,
        k
    )

    mrr = calculate_mrr(
        reference_answer,
        retrieved_documents,
        embedding_model
    )

    return {
        "recall_at_k": recall,
        "mrr": mrr
    }