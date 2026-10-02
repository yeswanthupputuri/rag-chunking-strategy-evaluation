from loaders.qa_loader import load_qa_file

from embeddings.embedding_model import (
    get_embedding_model
)

from rag.rag_pipeline import (
    answer_question
)

from evaluation.evaluator import (
    evaluate_result
)

QA_FILE = "data/questions.json"

COLLECTIONS = {
    "Fixed": "fixed_collection",
    "Recursive": "recursive_collection",
    "Semantic": "semantic_collection"
}

def evaluate_dataset():

    qa_data = load_qa_file(QA_FILE)

    embedding_model = get_embedding_model()

    all_results = []

    for question_number, item in enumerate(
        qa_data,
        start=1
    ):

        question = item["question"]
        reference_answer = item["answer"]

        print(
            f"QUESTION {question_number}"
        )

        print(f"\nQuestion: {question}")

        for strategy, collection_name in COLLECTIONS.items():

            print("\n----------------------------------------")
            print(f"Strategy: {strategy}")

            result = answer_question(
                question,
                collection_name
            )

            retrieved_documents = [
                item["document"]
                for item in result["retrieved_results"]
            ]

            evaluation = evaluate_result(
                question=question,
                generated_answer=result["answer"],
                reference_answer=reference_answer,
                retrieved_documents=retrieved_documents,
                embedding_model=embedding_model,
                retrieval_k=10
            )

            evaluation["strategy"] = strategy
            evaluation["question_number"] = question_number

            all_results.append(
                evaluation
            )

            print(
                f"\nGenerated Answer:\n"
                f"{evaluation['generated_answer']}"
            )

            print("\nMetrics:")

            print(
                f"Cosine Similarity: "
                f"{evaluation['cosine_similarity']:.4f}"
            )

            print(
                f"ROUGE-L: "
                f"{evaluation['rouge_l']:.4f}"
            )

    return all_results


if __name__ == "__main__":

    results = evaluate_dataset()
    print("========================================")
    print("DATASET EVALUATION COMPLETED")

    print(
        f"\nTotal evaluations: "
        f"{len(results)}"
    )