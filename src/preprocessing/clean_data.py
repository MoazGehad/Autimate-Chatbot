import json
import re
import os
import glob

def clean_arabic_text(text):
    if not isinstance(text, str):
        return text
    # Remove /uniXXXX and uniXXXX patterns
    text = re.sub(r'/uni[0-9A-F]{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'uni[0-9A-F]{4}', '', text, flags=re.IGNORECASE)
    # Remove Tatweel (Kashida)
    text = re.sub(r'[\u0640]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_english_text(text):
    if not isinstance(text, str):
        return text
    # Remove common footer/header patterns
    text = re.sub(r'Phone: 1300 308 699.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Email: amaze.org.au.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Approved \d{1,2}/\d{1,2}/\d{2,4}', '', text)
    text = re.sub(r'Page \d+', '', text)
    # Fix hyphenation (e.g. "communi- cation")
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_file(input_path, output_path, is_arabic=False):
    print(f"Processing {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = []
    for item in data:
        if 'content' in item:
            if is_arabic:
                item['content'] = clean_arabic_text(item['content'])
            else:
                item['content'] = clean_english_text(item['content'])
        
        # Also clean Q&A fields if they exist
        if 'question' in item:
            if is_arabic:
                item['question'] = clean_arabic_text(item['question'])
            else:
                item['question'] = clean_english_text(item['question'])
        if 'answer' in item:
            if is_arabic:
                item['answer'] = clean_arabic_text(item['answer'])
            else:
                item['answer'] = clean_english_text(item['answer'])
                
        cleaned_data.append(item)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

def main():
    input_dir = "processed/datasets"
    output_dir = "processed/cleaned_datasets"
    
    # Arabic files
    process_file(os.path.join(input_dir, "arabic_documents.json"), os.path.join(output_dir, "arabic_documents.json"), is_arabic=True)
    process_file(os.path.join(input_dir, "arabic_qa_combined.json"), os.path.join(output_dir, "arabic_qa_combined.json"), is_arabic=True)
    
    # English files
    process_file(os.path.join(input_dir, "english_documents.json"), os.path.join(output_dir, "english_documents.json"), is_arabic=False)
    process_file(os.path.join(input_dir, "english_qa_combined.json"), os.path.join(output_dir, "english_qa_combined.json"), is_arabic=False)
    
    # General conversations (treat as English/Mixed but mostly English cleaning rules apply or just whitespace)
    process_file(os.path.join(input_dir, "general_conversations.json"), os.path.join(output_dir, "general_conversations.json"), is_arabic=False)

if __name__ == "__main__":
    main()
