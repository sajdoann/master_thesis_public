"""
Script to create docs.jsonl from Wikipedia plaintext files and links.csv.
Creates documents in RAG format with docID, source (URL), and text fields.
"""

import csv
import json
import os
from pathlib import Path


def create_docs_jsonl(links_csv_path, plaintext_dir, output_path):
    """
    Create docs.jsonl file from links.csv and plaintext files.
    
    Args:
        links_csv_path: Path to links.csv file
        plaintext_dir: Directory containing plaintext .txt files
        output_path: Output path for docs.jsonl file
    """
    plaintext_dir = Path(plaintext_dir)
    output_path = Path(output_path)
    
    # Read links.csv to get docID, title, and url mappings
    docs_data = []
    with open(links_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = int(row['docID'])
            title = row['title'].strip()
            url = row['url'].strip()
            
            # Find corresponding plaintext file
            # Files are named like: {docID}_{title}.txt
            # We need to find the file that starts with {docID}_
            plaintext_file = None
            for txt_file in plaintext_dir.glob(f"{doc_id}_*.txt"):
                plaintext_file = txt_file
                break
            
            if plaintext_file is None or not plaintext_file.exists():
                print(f"Warning: Plaintext file not found for docID {doc_id} ({title})")
                continue
            
            # Read plaintext content
            try:
                with open(plaintext_file, 'r', encoding='utf-8') as txt_f:
                    text_content = txt_f.read().strip()
            except Exception as e:
                print(f"Error reading {plaintext_file}: {e}")
                continue
            
            # Format text according to the example format
            # Format: "title: {title}\nurl: {url}\ntext: {content}"
            formatted_text = f"title: {title}\nurl: {url}\ntext: {text_content}"
            
            # Create document entry
            doc_entry = {
                "docID": doc_id,
                "source": url,
                "text": formatted_text
            }
            
            docs_data.append(doc_entry)
    
    # Write to JSONL file (one JSON object per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in docs_data:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"Created {output_path} with {len(docs_data)} documents")


if __name__ == "__main__":
    # Set paths
    script_dir = Path(__file__).parent
    links_csv_path = script_dir / "wikipedia" / "links.csv"
    plaintext_dir = script_dir / "wikipedia" / "plaintext"
    output_path = script_dir / "wikipedia" / "docs.jsonl"
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create docs.jsonl
    create_docs_jsonl(links_csv_path, plaintext_dir, output_path)
