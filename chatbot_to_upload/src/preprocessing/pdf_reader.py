import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from cleaner import clean_text


def load_pdf(folder_path,language=None):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            for doc in loader.load():
                doc.page_content = clean_text(doc.page_content)
                if language:
                    doc.metadata["language"] = language
                documents.append(doc)

    return documents
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)
def docs_to_json(documents, output_folder, output_filename):
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, output_filename)

    data = []
    for document in documents:
        data.append({
            "content": document.page_content,
            "metadata": document.metadata
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_file}")


if __name__ == "__main__":

    arabic_folder = "../../datasets/arabic"
    english_folder = "../../datasets/english"
    output_folder = "../../processed/datasets"



    # Load and clean PDFs
    arabic_docs = load_pdf(arabic_folder, language="ar")
    english_docs = load_pdf(english_folder, language="en")

    # Split into chunks
    arabic_chunks = split_documents(arabic_docs)
    english_chunks = split_documents(english_docs)

    # Save to JSON
    docs_to_json(arabic_chunks, output_folder, "arabic_documents.json")
    docs_to_json(english_chunks, output_folder, "english_documents.json")

print("Arabic and English documents chunked and saved as JSON")