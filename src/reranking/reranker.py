from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"

class DocumentReranker:

    def __init__(self):

        self.model = CrossEncoder(
            RERANKER_MODEL
        )

    def rerank(
        self,
        question,
        documents,
        top_n=3
    ):

        if not documents:
            return []

        pairs = [
            [question, document.page_content]
            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        ranked_documents = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True
        )
        results = []

        for document, score in ranked_documents[:top_n]:
            results.append({
                "document": document,
                "score": float(score)
            })


        return results