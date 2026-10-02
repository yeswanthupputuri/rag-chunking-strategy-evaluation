from langchain_experimental.text_splitter import SemanticChunker

from embeddings.embedding_model import get_embedding_model

def semantic_chunk_documents(documents):
    embedding_model = get_embedding_model()

    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile"
    )

    chunks = splitter.split_documents(documents)

    return chunks