import json
from pathlib import Path


def load_qa_file(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Q&A file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as file:
        qa_data = json.load(file)

    if not isinstance(qa_data, list):
        raise ValueError("Q&A file must contain a JSON list.")

    for item in qa_data:
        if "question" not in item:
            raise ValueError("Every item must contain a 'question'.")

        if "answer" not in item:
            raise ValueError("Every item must contain an 'answer'.")

    return qa_data