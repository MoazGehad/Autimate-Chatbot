import os
import json

def docs_to_json(documents, output_folder, output_filename):
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, output_filename)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_file}")
