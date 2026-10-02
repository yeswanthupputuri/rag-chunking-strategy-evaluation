from langchain_text_splitters import CharacterTextSplitter

def fixed_chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=""
    )

    chunks = splitter.split_documents(documents)

    return chunks