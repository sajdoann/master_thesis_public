"""
Script to create a Wikipedia dataset from links.csv
Downloads HTML, converts to PDF, and extracts plaintext.
"""

import csv
import os
import re
import sys
import argparse
import requests
from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from weasyprint import HTML, CSS
from tqdm import tqdm
from PIL import Image
import io


def sanitize_filename(filename):
    """Convert a string to a safe filename."""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def sanitize_topic(topic):
    """Convert topic to filename-safe format without spaces and special characters."""
    # Remove or replace spaces and special characters with underscores
    # Keep alphanumeric characters and basic unicode characters
    topic = re.sub(r'[^\w\s-]', '_', topic)  # Replace special chars with underscore
    topic = re.sub(r'[\s]+', '_', topic)  # Replace spaces with underscore
    topic = re.sub(r'_+', '_', topic)  # Replace multiple underscores with single
    topic = topic.strip('_')  # Remove leading/trailing underscores
    # Limit length
    if len(topic) > 200:
        topic = topic[:200]
    return topic


def download_html(url, output_path):
    """Download HTML content from a URL and save it."""
    try:
        # Use desktop version of Wikipedia for better PDF conversion
        if 'wikipedia.org' in url:
            # Ensure we get the desktop version
            if '?mobileaction=' not in url:
                url = url.replace('m.wikipedia.org', 'cs.wikipedia.org')
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_plaintext(html_path, output_path):
    """Extract plaintext from HTML file, excluding Wikipedia UI elements."""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, and structural elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        
        # Remove Wikipedia-specific UI elements
        # Navigation and sidebars
        for element in soup.find_all(['div', 'aside'], class_=lambda x: x and any(
            cls in str(x).lower() for cls in ['nav', 'sidebar', 'navigation', 'mw-navigation', 'vector-sidebar']
        )):
            element.decompose()
        
        # Language selector
        for element in soup.find_all(['div', 'ul'], id=lambda x: x and 'p-lang' in str(x)):
            element.decompose()
        for element in soup.find_all(['div', 'ul'], class_=lambda x: x and 'p-lang' in str(x)):
            element.decompose()
        
        # Table of contents
        for element in soup.find_all(['div', 'ul'], id=lambda x: x and 'toc' in str(x)):
            element.decompose()
        for element in soup.find_all(['div', 'ul'], class_=lambda x: x and 'toc' in str(x)):
            element.decompose()
        
        # Edit links and Wikipedia UI chrome
        for element in soup.find_all(['span', 'a'], class_=lambda x: x and any(
            cls in str(x).lower() for cls in ['mw-editsection', 'mw-jump-link', 'mw-headline']
        )):
            element.decompose()
        
        # Remove edit section brackets [editovat | editovat zdroj]
        for element in soup.find_all(['span', 'a'], string=re.compile(r'\[.*edit.*\]', re.I)):
            element.decompose()
        
        # Remove references section navigation
        for element in soup.find_all(['div'], class_=lambda x: x and 'reflist' in str(x).lower()):
            # Keep the content but remove navigation links
            for link in element.find_all('a', class_=lambda x: x and 'mw-jump' in str(x)):
                link.decompose()
        
        # Remove infobox navigation and metadata
        for element in soup.find_all(['table'], class_=lambda x: x and 'infobox' in str(x).lower()):
            # Keep infobox but remove edit links
            for edit_link in element.find_all(['span', 'a'], class_=lambda x: x and 'edit' in str(x).lower()):
                edit_link.decompose()
        
        # Get main content area (Wikipedia-specific)
        content_div = soup.find('div', {'id': 'mw-content-text'})
        
        if not content_div:
            content_div = soup.find('main') or soup.find('body')
        
        if content_div:
            # Remove any remaining edit links and UI elements
            for edit_section in content_div.find_all(['span', 'a'], class_=lambda x: x and 'mw-editsection' in str(x)):
                edit_section.decompose()
            
            # Remove "editovat" links
            for link in content_div.find_all('a', string=re.compile(r'editovat|edit', re.I)):
                link.decompose()
            
            # Remove section edit brackets
            for span in content_div.find_all('span', string=re.compile(r'\[.*\]', re.I)):
                if 'edit' in span.get_text().lower():
                    span.decompose()
            
            text = content_div.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up Wikipedia-specific patterns
        # Remove standalone "[editovat | editovat zdroj]" lines
        text = re.sub(r'^\s*\[.*?edit.*?\]\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove multiple consecutive brackets (likely edit links)
        text = re.sub(r'\[\s*\]', '', text)
        
        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Final cleanup of multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Error extracting plaintext from {html_path}: {e}")
        return False


def clean_html_for_pdf(html_path):
    """Clean HTML file by removing Wikipedia UI elements, return cleaned HTML string."""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, and structural elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        
        # Remove Wikipedia-specific UI elements (same as plaintext extraction)
        for element in soup.find_all(['div', 'aside'], class_=lambda x: x and any(
            cls in str(x).lower() for cls in ['nav', 'sidebar', 'navigation', 'mw-navigation', 'vector-sidebar']
        )):
            element.decompose()
        
        # Language selector
        for element in soup.find_all(['div', 'ul'], id=lambda x: x and 'p-lang' in str(x)):
            element.decompose()
        for element in soup.find_all(['div', 'ul'], class_=lambda x: x and 'p-lang' in str(x)):
            element.decompose()
        
        # Table of contents
        for element in soup.find_all(['div', 'ul'], id=lambda x: x and 'toc' in str(x)):
            element.decompose()
        for element in soup.find_all(['div', 'ul'], class_=lambda x: x and 'toc' in str(x)):
            element.decompose()
        
        # Edit links and Wikipedia UI chrome
        for element in soup.find_all(['span', 'a'], class_=lambda x: x and any(
            cls in str(x).lower() for cls in ['mw-editsection', 'mw-jump-link', 'mw-headline']
        )):
            element.decompose()
        
        # Get main content area
        content_div = soup.find('div', {'id': 'mw-content-text'})
        
        if content_div:
            # Remove edit sections
            for edit_section in content_div.find_all(['span', 'a'], class_=lambda x: x and 'mw-editsection' in str(x)):
                edit_section.decompose()
        
        return str(soup)
    except Exception as e:
        print(f"Error cleaning HTML: {e}")
        return None


def html_to_pdf(html_path, output_path):
    """Convert HTML file to PDF, excluding Wikipedia UI elements."""
    try:
        # Clean the HTML first
        cleaned_html = clean_html_for_pdf(html_path)
        if cleaned_html is None:
            # Fallback to original if cleaning fails
            cleaned_html = open(html_path, 'r', encoding='utf-8').read()
        
        # Add basic CSS for better PDF rendering
        css_string = """
        @page {
            size: A4;
            margin: 2cm;
        }
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        .mw-parser-output {
            max-width: 100%;
        }
        /* Hide Wikipedia UI elements */
        .mw-editsection,
        .mw-jump-link,
        #p-lang,
        #toc,
        .toc,
        nav,
        header,
        footer,
        .vector-sidebar,
        .mw-navigation {
            display: none !important;
        }
        """
        css = CSS(string=css_string)
        HTML(string=cleaned_html).write_pdf(output_path, stylesheets=[css])
        return True
    except Exception as e:
        print(f"Error converting {html_path} to PDF: {e}")
        return False


def html_to_image_pdf(html_path, output_path):
    """Convert HTML file to image-based PDF (no text layer) for OCR.
    Tries Playwright first, falls back to pdf2image method if needed.
    """
    try:
        # Clean the HTML first
        cleaned_html = clean_html_for_pdf(html_path)
        if cleaned_html is None:
            # Fallback to original if cleaning fails
            cleaned_html = open(html_path, 'r', encoding='utf-8').read()
        
        # Add basic CSS for better PDF rendering
        css_string = """
        @page {
            size: A4;
            margin: 2cm;
        }
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        .mw-parser-output {
            max-width: 100%;
        }
        /* Hide Wikipedia UI elements */
        .mw-editsection,
        .mw-jump-link,
        #p-lang,
        #toc,
        .toc,
        nav,
        header,
        footer,
        .vector-sidebar,
        .mw-navigation {
            display: none !important;
        }
        """
        
        # Try Playwright first (better quality, handles full page)
        try:
            return html_to_image_pdf_playwright(html_path, output_path, cleaned_html, css_string)
        except Exception as e:
            # Fallback to pdf2image method
            return html_to_image_pdf_weasyprint_fallback(html_path, output_path, cleaned_html, css_string)
            
    except Exception as e:
        print(f"Error converting {html_path} to image PDF: {e}")
        return False


def html_to_image_pdf_playwright(html_path, output_path, cleaned_html, css_string):
    """Convert HTML to image-based PDF using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
        import tempfile
        
        # Create a temporary HTML file with cleaned content and CSS
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>{css_string}</style>
        </head>
        <body>
        {cleaned_html}
        </body>
        </html>
        """
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_html:
            tmp_html.write(full_html)
            tmp_html_path = tmp_html.name
        
        try:
            with sync_playwright() as p:
                # Launch browser
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Set viewport size for consistent rendering
                page.set_viewport_size({"width": 1200, "height": 1600})
                
                # Load the HTML file
                page.goto(f"file://{tmp_html_path}")
                
                # Wait for page to load
                page.wait_for_load_state("networkidle")
                
                # Take screenshot of the full page
                screenshot_bytes = page.screenshot(full_page=True, type='png')
                
                browser.close()
                
                # Convert screenshot to PDF
                img = Image.open(io.BytesIO(screenshot_bytes))
                
                # Convert to RGB if necessary (for PDF compatibility)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as PDF (image-based, no text layer)
                img.save(output_path, 'PDF', resolution=300.0)
                
                return True
        finally:
            # Clean up temp file
            os.unlink(tmp_html_path)
            
    except ImportError:
        # Playwright not available, try alternative method
        return html_to_image_pdf_weasyprint_fallback(html_path, output_path, cleaned_html, css_string)
    except Exception as e:
        print(f"Error with Playwright method: {e}")
        # Fallback
        return html_to_image_pdf_weasyprint_fallback(html_path, output_path, cleaned_html, css_string)


def html_to_image_pdf_weasyprint_fallback(html_path, output_path, cleaned_html, css_string):
    """Fallback method: Convert HTML to PDF, then PDF to images, then images to PDF.
    This creates an image-based PDF without text layer.
    """
    try:
        from pdf2image import convert_from_path
        import tempfile
        
        # First, create a regular PDF
        css = CSS(string=css_string)
        html_doc = HTML(string=cleaned_html)
        
        # Create temporary PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
            tmp_pdf_path = tmp_pdf.name
            html_doc.write_pdf(tmp_pdf_path, stylesheets=[css])
        
        # Convert PDF pages to images
        images = convert_from_path(tmp_pdf_path, dpi=300)
        
        # Convert images back to PDF (image-based, no text layer)
        if images:
            # Convert first image to RGB if needed
            rgb_images = []
            for img in images:
                if img.mode != 'RGB':
                    rgb_images.append(img.convert('RGB'))
                else:
                    rgb_images.append(img)
            
            # Save all pages as a single PDF
            rgb_images[0].save(
                output_path,
                'PDF',
                resolution=300.0,
                save_all=True,
                append_images=rgb_images[1:] if len(rgb_images) > 1 else []
            )
        
        # Clean up temp file
        os.unlink(tmp_pdf_path)
        
        return True
        
    except ImportError:
        print("Error: Need either 'playwright' or 'pdf2image' (with poppler) for image-based PDF creation")
        print("Install with: pip install playwright pdf2image")
        print("For playwright: playwright install chromium")
        print("For pdf2image: sudo apt-get install poppler-utils (on Ubuntu/Debian)")
        return False
    except Exception as e:
        print(f"Error with fallback method: {e}")
        return False


def pdf_to_image_pdf(pdf_path, output_path):
    """Convert a regular PDF to an image-based PDF (no text layer) for OCR.
    This function takes an existing PDF, converts each page to an image, 
    and then combines the images back into a PDF without a text layer.
    """
    try:
        from pdf2image import convert_from_path
        
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=300)
        
        if not images:
            print(f"Error: No pages found in PDF {pdf_path}")
            return False
        
        # Convert images to RGB if needed
        rgb_images = []
        for img in images:
            if img.mode != 'RGB':
                rgb_images.append(img.convert('RGB'))
            else:
                rgb_images.append(img)
        
        # Save all pages as a single PDF (image-based, no text layer)
        rgb_images[0].save(
            output_path,
            'PDF',
            resolution=300.0,
            save_all=True,
            append_images=rgb_images[1:] if len(rgb_images) > 1 else []
        )
        
        return True
        
    except ImportError:
        print("Error: Need 'pdf2image' (with poppler) for PDF to image-based PDF conversion")
        print("Install with: pip install pdf2image")
        print("For pdf2image: sudo apt-get install poppler-utils (on Ubuntu/Debian)")
        return False
    except Exception as e:
        print(f"Error converting PDF {pdf_path} to image PDF: {e}")
        return False


def process_wikipedia_links(csv_path, base_output_dir, force=False, create_image_pdfs=True):
    """Process all Wikipedia links from CSV file.
    
    Args:
        csv_path: Path to CSV file with Wikipedia links
        base_output_dir: Base directory for output files
        force: If True, re-download and regenerate all files
        create_image_pdfs: If True, also create image-based PDFs (no text layer) for OCR
    """
    # Create output directories
    html_dir = Path(base_output_dir) / 'html'
    pdf_dir = Path(base_output_dir) / 'pdf'
    plaintext_dir = Path(base_output_dir) / 'plaintext'
    image_pdf_dir = Path(base_output_dir) / 'pdf_images' if create_image_pdfs else None
    
    html_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    plaintext_dir.mkdir(parents=True, exist_ok=True)
    if image_pdf_dir:
        image_pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV file
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get('url', '').strip()]
    
    print(f"Processing {len(rows)} Wikipedia pages...")
    print(f"Output directories:")
    print(f"  HTML: {html_dir}")
    print(f"  PDF: {pdf_dir}")
    print(f"  Plaintext: {plaintext_dir}")
    if image_pdf_dir:
        print(f"  Image PDF (for OCR): {image_pdf_dir}")
    print()
    
    success_count = 0
    failed_count = 0
    
    for row in tqdm(rows, desc="Processing pages", unit="page"):
        doc_id = row.get('docID', '').strip()
        title = row['title'].strip()
        url = row['url'].strip()
        
        if not url:
            continue
        
        # Sanitize topic (title without spaces and special characters)
        sanitized_topic = sanitize_topic(title)
        
        # Create filename in format: {docID}_topic.ending
        filename_base = f"{doc_id}_{sanitized_topic}"
        
        # Define output paths
        html_path = html_dir / f"{filename_base}.html"
        pdf_path = pdf_dir / f"{filename_base}.pdf"
        plaintext_path = plaintext_dir / f"{filename_base}.txt"
        image_pdf_path = image_pdf_dir / f"{filename_base}.pdf" if create_image_pdfs else None
        
        # Skip if all files already exist (unless force is True)
        files_to_check = [html_path, pdf_path, plaintext_path]
        if create_image_pdfs and image_pdf_path:
            files_to_check.append(image_pdf_path)
        
        if not force and all(path.exists() for path in files_to_check):
            tqdm.write(f"✓ Skipping '{title}' (all files already exist)")
            success_count += 1
            continue
        
        # Download HTML
        if force or not html_path.exists():
            tqdm.write(f"Downloading HTML for '{title}'...")
            if not download_html(url, html_path):
                tqdm.write(f"✗ Failed to download HTML for '{title}'")
                failed_count += 1
                continue
            tqdm.write(f"  → Saved to: {html_path}")
        else:
            tqdm.write(f"  HTML already exists: {html_path}")
        
        # Extract plaintext
        if force or not plaintext_path.exists():
            tqdm.write(f"Extracting plaintext for '{title}'...")
            if not extract_plaintext(html_path, plaintext_path):
                tqdm.write(f"✗ Failed to extract plaintext for '{title}'")
                failed_count += 1
                continue
            tqdm.write(f"  → Saved to: {plaintext_path}")
        else:
            tqdm.write(f"  Plaintext already exists: {plaintext_path}")
        
        # Convert to PDF
        if force or not pdf_path.exists():
            tqdm.write(f"Converting to PDF for '{title}'...")
            if not html_to_pdf(html_path, pdf_path):
                tqdm.write(f"✗ Failed to convert PDF for '{title}'")
                failed_count += 1
                continue
            tqdm.write(f"  → Saved to: {pdf_path}")
        else:
            tqdm.write(f"  PDF already exists: {pdf_path}")
        
        # Create image-based PDF (no text layer) from regular PDF for OCR
        if create_image_pdfs and image_pdf_path:
            if force or not image_pdf_path.exists():
                # Ensure regular PDF exists before creating image PDF
                if not pdf_path.exists():
                    tqdm.write(f"✗ Cannot create image PDF: regular PDF does not exist for '{title}'")
                else:
                    tqdm.write(f"Creating image-based PDF from regular PDF for '{title}' (for OCR)...")
                    if not pdf_to_image_pdf(pdf_path, image_pdf_path):
                        tqdm.write(f"✗ Failed to create image PDF for '{title}'")
                        # Don't fail the whole process, just warn
                    else:
                        tqdm.write(f"  → Saved to: {image_pdf_path}")
            else:
                tqdm.write(f"  Image PDF already exists: {image_pdf_path}")
        
        tqdm.write(f"✓ Completed processing '{title}'")
        success_count += 1
    
    print(f"\nCompleted: {success_count} successful, {failed_count} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Wikipedia dataset from CSV links')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force re-download and regeneration of all files')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV file (default: wikipedia/links.csv)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: wikipedia/)')
    parser.add_argument('--no-image-pdfs', action='store_false', dest='image_pdfs', default=True,
                        help='Disable creation of image-based PDFs (no text layer) for OCR')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Set paths
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = script_dir / 'wikipedia' / 'links.csv'
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / 'wikipedia'
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    if args.force:
        print("Force mode: Re-downloading and regenerating all files...")
        print()
    
    # image_pdfs defaults to True, so only print if explicitly disabled
    if not args.image_pdfs:
        print("Image PDF creation disabled.")
        print()
    else:
        print("Image PDF mode: Creating image-based PDFs (no text layer) for OCR...")
        print()
    
    process_wikipedia_links(csv_path, output_dir, force=args.force, create_image_pdfs=args.image_pdfs)
    print(f"\nDataset location: {output_dir}")
    print(f"  HTML files: {output_dir / 'html'}")
    print(f"  PDF files: {output_dir / 'pdf'}")
    print(f"  Plaintext files: {output_dir / 'plaintext'}")
    if args.image_pdfs:
        print(f"  Image PDF files (for OCR): {output_dir / 'pdf_images'}")
