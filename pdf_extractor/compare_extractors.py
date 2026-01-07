"""
Compare PDF extractors: tika, docling, and PyPDFLoader.
Measures extraction time and character precision.
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
from Levenshtein import ratio as levenshtein_ratio

from tika import parser as tika_parser
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from langchain_community.document_loaders import PyPDFLoader


def extract_with_tika(pdf_path: Path, enable_ocr: bool = True) -> Tuple[Optional[str], float]:
    """Extract text using Apache Tika with optional OCR support.
    
    Args:
        pdf_path: Path to the PDF file
        enable_ocr: If True, enables OCR for image-based PDFs using Tesseract.
                    Requires Tesseract OCR to be installed and configured in Tika server.
    """
    if tika_parser is None:
        return None, 0.0
    
    try:
        start_time = time.time()
        
        # Configure headers for OCR if enabled
        headers = {}
        if enable_ocr:
            # Use ocr_and_text_extraction strategy: tries text extraction first,
            # then falls back to OCR for image-based PDFs
            headers['X-Tika-PDFOcrStrategy'] = 'ocr_and_text_extraction'
            # Optional: specify OCR language (default is 'eng' if not specified)
            # headers['X-Tika-OCRLanguage'] = 'eng'
        
        # Parse with optional OCR headers
        if headers:
            parsed = tika_parser.from_file(str(pdf_path), headers=headers)
        else:
            parsed = tika_parser.from_file(str(pdf_path))
        
        elapsed = time.time() - start_time
        
        if parsed and 'content' in parsed:
            text = parsed['content']
            return text if text else None, elapsed
        return None, elapsed
    except Exception as e:
        print(f"Error with tika on {pdf_path.name}: {e}")
        return None, 0.0


def extract_with_docling(pdf_path: Path) -> Tuple[Optional[str], float]:
    """Extract text using docling."""
    if DocumentConverter is None:
        return None, 0.0
    
    try:
        start_time = time.time()
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        elapsed = time.time() - start_time
        
        if result and hasattr(result, 'document'):
            doc = result.document
            if hasattr(doc, 'text'):
                return doc.text, elapsed
            elif hasattr(doc, 'export_to_markdown'):
                return doc.export_to_markdown(), elapsed
        return None, elapsed
    except Exception as e:
        print(f"Error with docling on {pdf_path.name}: {e}")
        return None, 0.0


def extract_with_pypdfloader(pdf_path: Path) -> Tuple[Optional[str], float]:
    """Extract text using PyPDFLoader from LangChain."""
    if PyPDFLoader is None:
        return None, 0.0
    
    try:
        start_time = time.time()
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        elapsed = time.time() - start_time
        
        if documents:
            # Combine all document pages into a single text
            text_parts = [doc.page_content for doc in documents if doc.page_content]
            extracted_text = "\n".join(text_parts)
            return extracted_text if extracted_text else None, elapsed
        return None, elapsed
    except Exception as e:
        print(f"Error with PyPDFLoader on {pdf_path.name}: {e}")
        return None, 0.0


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove extra whitespace)."""
    if not text:
        return ""
    # Convert to lowercase and remove extra whitespace
    normalized = " ".join(text.lower().split())
    return normalized


def calculate_chrf(
    extracted: Optional[str],
    reference: Optional[str]
) -> float:
    """Calculate character-level precision between extracted and reference text.
    
    Uses Hugging Face evaluate library's chrf metric for character-level F-score.
    Returns a score between 0 and 100 (chrf score ).
    """
    if extracted is None or reference is None:
        return 0.0
    
    if not extracted or not reference:
        return 0.0
    
    if evaluate is None:
        print("Warning: evaluate library not available. Install with: pip install evaluate")
        return 0.0
    
    # Load chrf metric
    chrf = evaluate.load("chrf")
    
    # chrf expects lists: predictions (list of str) and references (list of list of str)
    predictions = [extracted]
    references = [[reference]]
    
    # Compute chrf score
    results = chrf.compute(predictions=predictions, references=references)
    
    # chrf returns score between 0 and 100
    score = results['score'] 
    
    return score


def calculate_chrf_plus_plus(
    extracted: Optional[str],
    reference: Optional[str]
) -> float:
    """Calculate chrF++ score between extracted and reference text.
    
    Uses Hugging Face evaluate library's chrf metric with word_order=2 for chrF++.
    Returns a score between 0 and 100 (chrf++ score).
    """
    if extracted is None or reference is None:
        return 0.0
    
    if not extracted or not reference:
        return 0.0
    
    if evaluate is None:
        print("Warning: evaluate library not available. Install with: pip install evaluate")
        return 0.0
    
    # Load chrf metric
    chrf = evaluate.load("chrf")
    
    # chrf expects lists: predictions (list of str) and references (list of list of str)
    predictions = [extracted]
    references = [[reference]]
    
    # Compute chrF++ score with word_order=2
    results = chrf.compute(predictions=predictions, references=references, word_order=2)
    
    # chrf returns score between 0 and 100
    score = results['score'] 
    
    return score


def calculate_levenshtein_ratio(
    extracted: Optional[str],
    reference: Optional[str]
) -> float:
    """Calculate Levenshtein ratio between extracted and reference text.
    
    Uses Levenshtein.ratio from rapidfuzz/python-Levenshtein library.
    Returns a normalized similarity score between 0 and 1.0.
    """
    if extracted is None or reference is None:
        return 0.0
    
    if not extracted or not reference:
        return 0.0
    
    # Calculate normalized indel similarity (0 to 1.0)
    ratio = levenshtein_ratio(extracted, reference)
    return ratio


def calculate_char_diff(
    extracted: Optional[str],
    reference: Optional[str]
) -> int:
    """Calculate character count difference between extracted and reference text.
    
    Returns: extracted_char_count - reference_char_count
    Positive values indicate extracted text has more characters.
    Negative values indicate extracted text has fewer characters.
    """
    if extracted is None or reference is None:
        return 0
    
    extracted_chars = len(extracted) if extracted else 0
    reference_chars = len(reference) if reference else 0
    
    return extracted_chars - reference_chars


def calculate_char_diff_percentage(
    extracted: Optional[str],
    reference: Optional[str]
) -> float:
    """Calculate character count difference as a percentage of reference text.
    
    Returns: ((extracted_char_count - reference_char_count) / reference_char_count) * 100
    Positive values indicate extracted text has more characters (as % of reference).
    Negative values indicate extracted text has fewer characters (as % of reference).
    Returns 0.0 if reference is empty or None.
    """
    if extracted is None or reference is None:
        return 0.0
    
    extracted_chars = len(extracted) if extracted else 0
    reference_chars = len(reference) if reference else 0
    
    if reference_chars == 0:
        return 0.0
    
    diff = extracted_chars - reference_chars
    percentage = (diff / reference_chars) * 100.0
    
    return percentage


def load_reference_text(plaintext_path: Path) -> Optional[str]:
    """Load reference plaintext file."""
    try:
        with open(plaintext_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading reference text from {plaintext_path.name}: {e}")
        return None


def save_extracted_text(
    text: Optional[str],
    output_dir: Path,
    extractor_name: str,
    pdf_name: str
) -> None:
    """Save extracted text to a file."""
    if text is None:
        return
    
    extractor_dir = output_dir / extractor_name
    extractor_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename from PDF name
    output_filename = pdf_name.replace('.pdf', '.txt')
    output_path = extractor_dir / output_filename
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Warning: Failed to save extracted text to {output_path}: {e}")


def compare_extractors(
    pdf_dir: Path,
    plaintext_dir: Path,
    output_path: Path,
    max_docs: Optional[int] = None,
    save_outputs: bool = True,
    pdf_dir_name: Optional[str] = None,
    enable_tika_ocr: bool = True
) -> None:
    """Compare PDF extractors on a set of PDF files.
    
    Args:
        pdf_dir: Directory containing PDF files
        plaintext_dir: Directory containing reference plaintext files
        output_path: Path to output CSV file
        max_docs: Maximum number of documents to process
        save_outputs: If True, save extracted texts to files
        pdf_dir_name: Optional name for organizing extracted texts in subdirectories
        enable_tika_ocr: If True, enables OCR for Tika extractor (requires Tesseract in Tika server)
    """
    
    # Check which extractors are available
    extractors = {}
    if tika_parser is not None:
        # Wrap tika extractor to pass enable_ocr parameter
        extractors['tika'] = lambda path: extract_with_tika(path, enable_ocr=enable_tika_ocr)
    if DocumentConverter is not None:
        extractors['docling'] = extract_with_docling
        extractors['pypdfloader'] = extract_with_pypdfloader
    
    if not extractors:
        print("Error: No PDF extractors available. Please install tika, docling, or langchain-community.")
        return
    
    print(f"Available extractors: {', '.join(extractors.keys())}")
    if 'tika' in extractors and enable_tika_ocr:
        print("Tika OCR is ENABLED (requires Tesseract in Tika server)")
    elif 'tika' in extractors:
        print("Tika OCR is DISABLED")
    
    # Create output directory for extracted texts if needed
    extracted_texts_dir = None
    if save_outputs:
        extracted_texts_dir = output_path.parent / 'extracted_texts'
        if pdf_dir_name:
            extracted_texts_dir = extracted_texts_dir / pdf_dir_name
        extracted_texts_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracted texts will be saved to: {extracted_texts_dir}")
    
    # Get all PDF files
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    if max_docs:
        pdf_files = pdf_files[:max_docs]
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    results: List[Dict[str, Any]] = []
    
    for pdf_path in pdf_files:
        pdf_name = pdf_path.name
        plaintext_path = plaintext_dir / pdf_name.replace('.pdf', '.txt')
        
        if not plaintext_path.exists():
            print(f"Warning: No reference text found for {pdf_name}, skipping...")
            continue
        
        reference_text = load_reference_text(plaintext_path)
        if not reference_text:
            continue
        
        result = {
            'pdf_file': pdf_name,
            'reference_chars': len(reference_text)
        }
        
        # Extract with each available extractor
        for extractor_name, extract_func in extractors.items():
            extracted_text, elapsed_time = extract_func(pdf_path)
            
            result[f'{extractor_name}_time'] = f"{elapsed_time:.4f}"
            result[f'{extractor_name}_chars'] = len(extracted_text) if extracted_text else 0
            
            if extracted_text:
                chrf_score = calculate_chrf(extracted_text, reference_text)
                result[f'{extractor_name}_chrf'] = f"{chrf_score:.4f}"
                
                chrf_plus_plus_score = calculate_chrf_plus_plus(extracted_text, reference_text)
                result[f'{extractor_name}_chrfPlusPlus'] = f"{chrf_plus_plus_score:.4f}"
                
                levenshtein_ratio_score = calculate_levenshtein_ratio(extracted_text, reference_text)
                result[f'{extractor_name}_levenshteinRatio'] = f"{levenshtein_ratio_score:.4f}"
                
                char_diff = calculate_char_diff(extracted_text, reference_text)
                result[f'{extractor_name}_charDiff'] = str(char_diff)
                
                char_diff_pct = calculate_char_diff_percentage(extracted_text, reference_text)
                result[f'{extractor_name}_charDiffPct'] = f"{char_diff_pct:.4f}"

                
                # Save extracted text to file
                if save_outputs and extracted_texts_dir:
                    save_extracted_text(
                        extracted_text,
                        extracted_texts_dir,
                        extractor_name,
                        pdf_name
                    )
            else:
                result[f'{extractor_name}_chrf'] = "0.0000"
                result[f'{extractor_name}_chrfPlusPlus'] = "0.0000"
                result[f'{extractor_name}_levenshteinRatio'] = "0.0000"
                result[f'{extractor_name}_charDiff'] = "0"
                result[f'{extractor_name}_charDiffPct'] = "0.0000"
        
        results.append(result)
        print(f"Processed: {pdf_name}")
    
    # Write results to CSV
    if results:
        fieldnames = ['pdf_file', 'reference_chars']
        for extractor_name in extractors.keys():
            fieldnames.extend([
                f'{extractor_name}_time',
                f'{extractor_name}_chars',
                f'{extractor_name}_chrf',
                f'{extractor_name}_chrfPlusPlus',
                f'{extractor_name}_levenshteinRatio',
                f'{extractor_name}_charDiff',
                f'{extractor_name}_charDiffPct'
            ])
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Processed {len(results)} PDF files.")
    else:
        print("No results to save.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare PDF extractors (tika, docling, PyPDFLoader)'
    )
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default=None,
        help='Directory containing PDF files (default: wikipedia/pdf)'
    )
    parser.add_argument(
        '--plaintext-dir',
        type=str,
        default=None,
        help='Directory containing reference plaintext files (default: wikipedia/plaintext)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: results/extractor_comparison.csv)'
    )
    parser.add_argument(
        '--max-docs',
        type=int,
        default=None,
        help='Maximum number of documents to process'
    )
    parser.add_argument(
        '--no-save-outputs',
        action='store_true',
        help='Do not save extracted text outputs to files'
    )
    parser.add_argument(
        '--process-both',
        action='store_true',
        help='Process both pdf and pdf_images directories, generating separate results for each'
    )
    parser.add_argument(
        '--disable-tika-ocr',
        action='store_true',
        help='Disable OCR for Tika extractor. By default, OCR is enabled for image-based PDFs.'
    )
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Set default paths
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
    else:
        pdf_dir = script_dir / 'wikipedia' / 'pdf'
    
    if args.plaintext_dir:
        plaintext_dir = Path(args.plaintext_dir)
    else:
        plaintext_dir = script_dir / 'wikipedia' / 'plaintext'
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / 'results' / 'extractor_comparison.csv'
    
    if not plaintext_dir.exists():
        print(f"Error: Plaintext directory not found: {plaintext_dir}")
        return
    
    # Handle --process-both flag
    if args.process_both:
        # Process both pdf and pdf_images directories
        pdf_dirs = [
            ('pdf', script_dir / 'wikipedia' / 'pdf'),
            ('pdf_images', script_dir / 'wikipedia' / 'pdf_images')
        ]
        
        for dir_name, dir_path in pdf_dirs:
            if not dir_path.exists():
                print(f"Warning: Directory not found: {dir_path}, skipping...")
                continue
            
            # Generate output path with directory name
            output_dir = output_path.parent
            output_filename = output_path.stem + f'_{dir_name}' + output_path.suffix
            dir_output_path = output_dir / output_filename
            
            print(f"\n{'='*60}")
            print(f"Processing directory: {dir_name}")
            print(f"{'='*60}")
            
            compare_extractors(
                dir_path,
                plaintext_dir,
                dir_output_path,
                args.max_docs,
                save_outputs=not args.no_save_outputs,
                pdf_dir_name=dir_name,
                enable_tika_ocr=not args.disable_tika_ocr
            )
    else:
        # Single directory processing (existing behavior)
        if not pdf_dir.exists():
            print(f"Error: PDF directory not found: {pdf_dir}")
            return
        
        # Use directory name from the pdf_dir path for organizing outputs
        pdf_dir_name = pdf_dir.name
        
        compare_extractors(
            pdf_dir,
            plaintext_dir,
            output_path,
            args.max_docs,
            save_outputs=not args.no_save_outputs,
            pdf_dir_name=pdf_dir_name,
            enable_tika_ocr=not args.disable_tika_ocr
        )


if __name__ == "__main__":
    main()

