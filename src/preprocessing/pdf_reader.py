import re

def clean_arabic_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Remove /uniXXXX and uniXXXX patterns
    text = re.sub(r'/uni[0-9A-F]{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'uni[0-9A-F]{4}', '', text, flags=re.IGNORECASE)
    # Remove Tatweel (Kashida)
    text = re.sub(r'[\u0640]', '', text)
    
    # Normalize Alifs
    text = re.sub(r'[أإآ]', 'ا', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text: str) -> str:
    """General cleaning with specific Arabic handling if detected."""
    # Basic cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Check if likely Arabic (simple heuristic)
    if re.search(r'[\u0600-\u06FF]', text):
        text = clean_arabic_text(text)
        
    return text

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
        separators=["\n\n", "\n", ".\s", "؟\s", "!\s", "،", " ", ""]
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