from evaluation.answer_metrics import (
    evaluate_answer
)

from evaluation.retrieval_metrics import (
    evaluate_retrieval
)


def evaluate_result(
    question,
    generated_answer,
    reference_answer,
    retrieved_documents,
    embedding_model,
    retrieval_k=10
):
    """
    Evaluate one RAG result.

    Evaluates:
    1. Retrieval Recall@K
    2. Retrieval MRR
    3. Answer Cosine Similarity
    4. Answer ROUGE-L
    """

    retrieval_results = evaluate_retrieval(
        reference_answer=reference_answer,
        retrieved_documents=retrieved_documents,
        embedding_model=embedding_model,
        k=retrieval_k
    )

    answer_results = evaluate_answer(
        generated_answer=generated_answer,
        reference_answer=reference_answer,
        embedding_model=embedding_model
    )

    return {
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,

        "recall_at_k": retrieval_results[
            "recall_at_k"
        ],

        "mrr": retrieval_results[
            "mrr"
        ],

        "cosine_similarity": answer_results[
            "cosine_similarity"
        ],

        "rouge_l": answer_results[
            "rouge_l"
        ]
    }