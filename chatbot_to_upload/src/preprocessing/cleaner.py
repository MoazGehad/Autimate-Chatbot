import re

def clean_text(text: str) -> str:
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove emails
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    # Remove DOIs or long IDs
    text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", "", text, flags=re.I)
    # Remove special characters
    text = re.sub(r"[©®™•→⇔►■□\-–—]", " ", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip very short lines
        if len(line) < 3:
            continue
        if re.match(r"^(صفحة|ص|Page|Chapter|Table|Figure|References)", line, re.I):
            continue
        if re.match(r"^(الملحق|الشكل|الجدول)", line):
            continue
        cleaned.append(line)

    return re.sub(r"\s+", " ", " ".join(cleaned)).strip()

