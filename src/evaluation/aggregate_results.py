import pandas as pd

def aggregate_results(results):
    df = pd.DataFrame(results)
    summary = (
        df.groupby("strategy")
        .agg(
            average_cosine_similarity=(
                "cosine_similarity", "mean"
            ),
            average_rouge_l=(
                "rouge_l", "mean"
            )
        )
        .reset_index()
    )

    return summary