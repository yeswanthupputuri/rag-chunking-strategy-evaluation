from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer


def calculate_cosine_similarity(
    generated_answer,
    reference_answer,
    embedding_model
):
    if not generated_answer:
        return 0.0

    if not reference_answer:
        return 0.0

    generated_embedding = embedding_model.embed_query(
        generated_answer
    )

    reference_embedding = embedding_model.embed_query(
        reference_answer
    )

    score = cosine_similarity(
        [generated_embedding],
        [reference_embedding]
    )[0][0]

    return float(score)


def calculate_rouge_l(
    generated_answer,
    reference_answer
):
    if not generated_answer:
        return 0.0

    if not reference_answer:
        return 0.0

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(
        reference_answer,
        generated_answer
    )

    return float(
        scores["rougeL"].fmeasure
    )


def evaluate_answer(
    generated_answer,
    reference_answer,
    embedding_model
):
    cosine_score = calculate_cosine_similarity(
        generated_answer,
        reference_answer,
        embedding_model
    )

    rouge_l_score = calculate_rouge_l(
        generated_answer,
        reference_answer
    )

    return {
        "cosine_similarity": cosine_score,
        "rouge_l": rouge_l_score
    }