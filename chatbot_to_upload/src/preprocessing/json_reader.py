import os
import json
from io_utils import docs_to_json  # Function to save JSON

def normalize_item(item, source=None,language=None):
    item_type = item.get("type", "qa")
    source_name = source or "unknown"

    # QA type
    if "question" in item and "answer" in item:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if not question or not answer:
            return None
        content = f"Question: {question} Answer: {answer}"

    # Info or tips type
    elif "title" in item or "content" in item:
        title = item.get("title", "").strip()
        body = item.get("content", "").strip()
        if title and body:
            content = f"{title}: {body}"
        elif body:
            content = body
        elif title:
            content = title
        else:
            return None

    else:
        texts = [str(v).strip() for v in item.values() if isinstance(v, str) and v.strip()]
        if not texts:
            return None
        content = " | ".join(texts)

    return {
        "content": content,
        "metadata": {"type": item_type, "source": source_name, "language": language or "unknown"}
    }

def load_qa(folder_path, source_mapping=None,language=None):

    if source_mapping is None:
        source_mapping = {}

    all_items = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)
        source_name = source_mapping.get(filename, None)

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {file_path} is not valid JSON, skipping.")
                continue

            for item in data:
                normalized = normalize_item(item, source_name,language)
                if normalized:
                    all_items.append(normalized)

    return all_items


if __name__ == "__main__":
    # Paths
    arabic_folder = "../../datasets/arabic"
    english_folder = "../../datasets/english"
    output_folder = "../../processed/datasets"

    qa_sources_arabic = {
        "autism_dataset.json": "aljazeera",
        "autism_dataset2.json": "mayo_clinic",
        "autism_dataset3.json": "altibbi",
        "autism_dataset4.json": "who",
        "autism_dataset5.json": "moh_sa"
    }

    # Load Q&A
    arabic_qa = load_qa(arabic_folder, qa_sources_arabic, language="ar")
    english_qa = load_qa(english_folder, language="en")


    # Save to JSON
    docs_to_json(arabic_qa, output_folder, "arabic_qa_combined.json")
    docs_to_json(english_qa, output_folder, "english_qa_combined.json")

    print(f"Arabic: {len(arabic_qa)}, English: {len(english_qa)} items combined and saved.")
