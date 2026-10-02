from rag.rag_pipeline import (
    answer_question
)

question = "What is the main objective of the document?"

collections = {
    "Fixed": "fixed_collection",
    "Recursive": "recursive_collection",
    "Semantic": "semantic_collection"
}


for strategy, collection_name in collections.items():

    print("\n")
    print("========================================")
    print(strategy)

    result = answer_question(
        question, collection_name
    )

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nRetrieved Chunks:")

    for index, item in enumerate(
        result["retrieved_results"]
    ):

        document = item["document"]
        score = item["score"]

        print(
            f"\nRank {index + 1}"
        )

        print(
            f"Reranker Score: {score}"
        )
        print(
            document.page_content[:300]
        )