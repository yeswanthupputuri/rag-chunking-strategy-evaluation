import sys
from pathlib import Path

import streamlit as st
import pandas as pd

SRC_PATH = Path(__file__).parent / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# Loaders
from loaders.document_loader import load_document
from loaders.qa_loader import load_qa_file

# Chunking
from chunking.fixed_chunking import fixed_chunk_documents
from chunking.recursive_chunking import recursive_chunk_documents
from chunking.semantic_chunking import semantic_chunk_documents

# Embeddings
from embeddings.embedding_model import get_embedding_model

# Vector store
from vectorstore.chroma_manager import (
    index_documents,
    clear_collection
)

# RAG
from rag.rag_pipeline import answer_question

# Evaluation
from evaluation.answer_metrics import evaluate_answer

COLLECTIONS = {
    "Fixed": "fixed_collection",
    "Recursive": "recursive_collection",
    "Semantic": "semantic_collection"
}

st.set_page_config(
    page_title="RAG Chunking Evaluation",
    page_icon="📚",
    layout="wide"
)


st.title("📚 RAG Chunking Evaluation")

st.write(
    "Compare Fixed, Recursive, and Semantic chunking "
    "using the same retrieval, reranking, LLM, and "
    "evaluation pipeline."
)

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "qa_data" not in st.session_state:
    st.session_state.qa_data = None

st.header("1. Upload Document")

uploaded_document = st.file_uploader(
    "Upload PDF, TXT, or DOCX",
    type=["pdf", "txt", "docx"]
)


if uploaded_document is not None:

    st.write(
        f"Selected document: "
        f"**{uploaded_document.name}**"
    )

    if st.button("Process Document"):

        data_directory = Path("data")
        data_directory.mkdir(exist_ok=True)

        document_path = (
            data_directory /
            uploaded_document.name
        )

        with open(document_path, "wb") as file:
            file.write(
                uploaded_document.getbuffer()
            )

        with st.spinner(
            "Loading and processing document..."
        ):

            # Load document
            documents = load_document(
                str(document_path)
            )

            st.write(
                f"Loaded {len(documents)} document sections."
            )

            # Create embeddings
            embedding_model = (
                get_embedding_model()
            )

            # Clear old collections
            for collection_name in COLLECTIONS.values():

                try:
                    clear_collection(
                        collection_name
                    )
                except Exception:
                    pass

            with st.spinner(
                "Creating Fixed chunks..."
            ):

                fixed_chunks = (
                    fixed_chunk_documents(
                        documents
                    )
                )

                index_documents(
                    fixed_chunks,
                    "fixed_collection"
                )

            with st.spinner(
                "Creating Recursive chunks..."
            ):

                recursive_chunks = (
                    recursive_chunk_documents(
                        documents
                    )
                )

                index_documents(
                    recursive_chunks,
                    "recursive_collection"
                )

            with st.spinner(
                "Creating Semantic chunks..."
            ):

                semantic_chunks = (
                    semantic_chunk_documents(
                        documents
                    )
                )

                index_documents(
                    semantic_chunks,
                    "semantic_collection"
                )

        st.session_state.document_processed = True

        st.success(
            "Document processed and all three "
            "vector collections created successfully."
        )

        # Show chunk counts
        st.subheader("Chunk Statistics")

        chunk_data = pd.DataFrame(
            {
                "Strategy": [
                    "Fixed",
                    "Recursive",
                    "Semantic"
                ],
                "Number of Chunks": [
                    len(fixed_chunks),
                    len(recursive_chunks),
                    len(semantic_chunks)
                ]
            }
        )

        st.dataframe(
            chunk_data,
            use_container_width=True
        )


st.header("2. Load Q&A Dataset")

qa_file = st.file_uploader(
    "Upload Q&A JSON file",
    type=["json"]
)


if qa_file is not None:

    try:

        qa_data = qa_file.getvalue().decode(
            "utf-8"
        )

        import json

        parsed_qa = json.loads(qa_data)

        if not isinstance(parsed_qa, list):
            st.error(
                "Q&A file must contain a JSON list."
            )

        else:

            st.session_state.qa_data = (
                parsed_qa
            )

            st.success(
                f"Loaded {len(parsed_qa)} questions."
            )

    except Exception as error:

        st.error(
            f"Error loading Q&A file: {error}"
        )


elif Path("data/questions.json").exists():

    if st.session_state.qa_data is None:

        try:
            st.session_state.qa_data = (
                load_qa_file(
                    "data/questions.json"
                )
            )

        except Exception as error:

            st.error(
                f"Could not load default Q&A file: "
                f"{error}"
            )


if (
    st.session_state.document_processed
    and st.session_state.qa_data
):

    st.header("3. Evaluation")

    evaluation_mode = st.radio(
        "Select evaluation mode",
        [
            "Single Question",
            "Full Dataset"
        ]
    )

    if evaluation_mode == "Single Question":

        questions = st.session_state.qa_data

        question_options = [
            f"{index + 1}. {item['question']}"
            for index, item in enumerate(questions)
        ]

        selected_question = st.selectbox(
            "Select a question",
            question_options
        )

        question_number = (
            question_options.index(
                selected_question
            )
        )

        selected_item = questions[
            question_number
        ]

        question = selected_item["question"]
        reference_answer = selected_item["answer"]

        st.subheader("Question")

        st.write(question)

        st.subheader("Reference Answer")

        st.info(reference_answer)

        if st.button(
            "Run Single Question Evaluation"
        ):

            embedding_model = (
                get_embedding_model()
            )

            results = []

            for strategy, collection_name in (
                COLLECTIONS.items()
            ):

                st.subheader(
                    f"{strategy} Strategy"
                )

                with st.spinner(
                    f"Generating answer using "
                    f"{strategy}..."
                ):

                    result = answer_question(
                        question,
                        collection_name
                    )

                    metrics = evaluate_answer(
                        generated_answer=result[
                            "answer"
                        ],
                        reference_answer=(
                            reference_answer
                        ),
                        embedding_model=(
                            embedding_model
                        )
                    )

                st.write(
                    "**Generated Answer:**"
                )

                st.write(
                    result["answer"]
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Cosine Similarity",
                        f"{metrics['cosine_similarity']:.4f}"
                    )

                with col2:
                    st.metric(
                        "ROUGE-L",
                        f"{metrics['rouge_l']:.4f}"
                    )

                results.append(
                    {
                        "Strategy": strategy,
                        "Cosine Similarity": (
                            metrics[
                                "cosine_similarity"
                            ]
                        ),
                        "ROUGE-L": (
                            metrics["rouge_l"]
                        )
                    }
                )

            st.subheader(
                "Strategy Comparison"
            )

            comparison_df = pd.DataFrame(
                results
            )

            st.dataframe(
                comparison_df,
                use_container_width=True
            )

    else:

        st.write(
            f"Questions available: "
            f"**{len(st.session_state.qa_data)}**"
        )

        if st.button(
            "Run Full Dataset Evaluation"
        ):

            embedding_model = (
                get_embedding_model()
            )

            all_results = []

            total_questions = len(
                st.session_state.qa_data
            )

            total_runs = (
                total_questions *
                len(COLLECTIONS)
            )

            progress = st.progress(0)

            completed = 0

            for question_number, item in enumerate(
                st.session_state.qa_data,
                start=1
            ):

                question = item["question"]
                reference_answer = item["answer"]

                for strategy, collection_name in (
                    COLLECTIONS.items()
                ):

                    with st.spinner(
                        f"Question {question_number}/"
                        f"{total_questions} - "
                        f"{strategy}"
                    ):

                        result = answer_question(
                            question,
                            collection_name
                        )

                        metrics = evaluate_answer(
                            generated_answer=(
                                result["answer"]
                            ),
                            reference_answer=(
                                reference_answer
                            ),
                            embedding_model=(
                                embedding_model
                            )
                        )

                    all_results.append(
                        {
                            "Question Number": (
                                question_number
                            ),
                            "Strategy": strategy,
                            "Cosine Similarity": (
                                metrics[
                                    "cosine_similarity"
                                ]
                            ),
                            "ROUGE-L": (
                                metrics["rouge_l"]
                            )
                        }
                    )

                    completed += 1

                    progress.progress(
                        completed / total_runs
                    )

            st.success(
                "Full dataset evaluation completed."
            )

            results_df = pd.DataFrame(
                all_results
            )

            st.subheader(
                "Detailed Results"
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            st.subheader(
                "Aggregated Results"
            )

            summary = (
                results_df
                .groupby("Strategy")
                .agg(
                    Average_Cosine_Similarity=(
                        "Cosine Similarity",
                        "mean"
                    ),
                    Average_ROUGE_L=(
                        "ROUGE-L",
                        "mean"
                    )
                )
                .reset_index()
            )

            st.dataframe(
                summary,
                use_container_width=True
            )

else:

    st.info(
        "Upload and process a document first, "
        "then load the Q&A dataset to begin evaluation."
    )