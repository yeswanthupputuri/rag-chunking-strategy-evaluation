from loaders.qa_loader import load_qa_file
from embeddings.embedding_model import get_embedding_model
from rag.rag_pipeline import answer_question
from evaluation.answer_metrics import evaluate_answer

QA_FILE = "data/questions.json"

QUESTION_NUMBER = 1

COLLECTIONS = {
    "Fixed": "fixed_collection",
    "Recursive": "recursive_collection",
    "Semantic": "semantic_collection"
}

def evaluate_single_question():
    qa_data = load_qa_file(QA_FILE)

    if QUESTION_NUMBER < 1 or QUESTION_NUMBER > len(qa_data):
        raise ValueError(
            f"Question number must be between 1 and {len(qa_data)}."
        )

    item = qa_data[QUESTION_NUMBER - 1]

    question = item["question"]
    reference_answer = item["answer"]

    embedding_model = get_embedding_model()

    print("\n")
    print("========================================")
    print("SINGLE QUESTION EVALUATION")

    print(f"\nQuestion Number: {QUESTION_NUMBER}")
    print(f"\nQuestion:\n{question}")

    print("\n----------------------------------------")
    print("REFERENCE ANSWER")
    print(reference_answer)

    print("\n")
    print("========================================")
    print("STRATEGY COMPARISON")

    results = []

    for strategy, collection_name in COLLECTIONS.items():

        print("\n")
        print("----------------------------------------")
        print(f"Strategy: {strategy}")

        result = answer_question(
            question,
            collection_name
        )

        generated_answer = result["answer"]

        metrics = evaluate_answer(
            generated_answer=generated_answer,
            reference_answer=reference_answer,
            embedding_model=embedding_model
        )

        result_data = {
            "strategy": strategy,
            "generated_answer": generated_answer,
            "cosine_similarity": metrics["cosine_similarity"],
            "rouge_l": metrics["rouge_l"]
        }

        results.append(result_data)

        print("\nGenerated Answer:")
        print(generated_answer)

        print("\nMetrics:")

        print(
            f"Cosine Similarity: "
            f"{metrics['cosine_similarity']:.4f}"
        )

        print(
            f"ROUGE-L: "
            f"{metrics['rouge_l']:.4f}"
        )

    return results


if __name__ == "__main__":
    evaluate_single_question()