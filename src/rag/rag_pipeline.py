from retrieval.retriever import (
    retrieve_and_rerank
)

from llm.huggingface_llm import (
    generate_answer
)

def build_context(ranked_results):
    context_parts = []

    for result in ranked_results:
        document = result["document"]
        context_parts.append(
            document.page_content
        )

    return "\n\n".join(
        context_parts
    )


def answer_question(
    question, collection_name
):

    ranked_results = retrieve_and_rerank(
        question, collection_name
    )

    context = build_context(
        ranked_results
    )

    answer = generate_answer(
        question, context
    )

    return {
        "question": question,
        "answer": answer,
        "retrieved_results": ranked_results,
        "context": context
    }