# Dataset Creation Methodology: Wikipedia Articles on Informatics and Natural Language Processing

## Overview

This dataset consists of 20 Wikipedia articles in Czech language covering topics in informatics and natural language processing. The articles were selected to provide a general knowledge base for testing a Retrieval-Augmented Generation (RAG) pipeline with a small group of students. The topics include natural language processing, information retrieval, lemmatization, corpus linguistics, machine learning, and related concepts. While Wikipedia is a commonly sourced reference, it is important to remain vigilant about potential misinformation.

## Data Collection Process

### Manual URL Selection

A curated list of Wikipedia URLs was manually compiled, focusing on Czech-language articles (cs.wikipedia.org) relevant to informatics and natural language processing. Each URL was assigned a unique document identifier (docID) ranging from 0 to 19. The selected topics include:

- Artificial intelligence (umělá inteligence)
- Machine learning (strojové učení)
- Algorithms (algoritmus)
- Natural language processing (zpracování přirozeného jazyka)
- Word2Vec
- Machine translation (strojový překlad)
- ChatGPT
- Neural networks (neuronová síť)
- Hidden Markov models (skrytý markovův model)
- Information retrieval (získávání informací)
- Text mining (dolování z textů)
- Lemmatization (lemmatizace)
- Language corpus (jazykový korpus)
- And related topics

The URLs were stored in a CSV file (`links.csv`) with three columns: `docID`, `title`, and `url`. This manual curation process ensured that only relevant, high-quality articles were included in the dataset.

### Rationale for Source Selection

Wikipedia articles were chosen as the primary data source because they provide:
1. **General accessibility**: Wikipedia is a commonly referenced source, making it suitable for testing RAG systems in realistic scenarios
2. **Structured content**: Articles follow consistent formatting, facilitating automated processing
3. **Comprehensive coverage**: Articles cover fundamental concepts in informatics and NLP
4. **Czech language content**: The articles are in Czech, providing a non-English language dataset for testing

Alternative sources, such as lecture materials from UFAL (Ústav formální a aplikované lingvistiky), were considered but deemed unsuitable due to their presentation-heavy format, reliance on visual context (images and diagrams), and inclusion of extensive programming examples that would not translate well to text-based retrieval systems.

## Web Scraping and Data Extraction

### HTML Download

The scraping process was implemented using Python scripts (`create_wikipedia_dataset.py`). For each URL in the CSV file, the following steps were performed:

1. **HTTP Request**: The script sends an HTTP GET request to the Wikipedia URL using the `requests` library with appropriate headers:
   - User-Agent header set to mimic a standard web browser (Mozilla/5.0)
   - Desktop version of Wikipedia was ensured (converting mobile URLs to desktop versions)
   - UTF-8 encoding was detected and applied

2. **HTML Storage**: The raw HTML content was saved to disk in the `html/` directory with filenames following the pattern `{docID}_{sanitized_topic}.html`, where the topic name was sanitized to be filesystem-safe (spaces and special characters replaced with underscores).

### Plaintext Extraction

Plaintext extraction was performed using BeautifulSoup4, a Python HTML parsing library. The extraction process involved:

1. **HTML Parsing**: The HTML content was parsed into a structured document object model (DOM)

2. **Content Filtering**: Wikipedia-specific UI elements were removed to extract only the article content:
   - Navigation elements (`nav`, `header`, `footer`)
   - Sidebars and navigation menus (classes containing 'nav', 'sidebar', 'navigation', 'mw-navigation', 'vector-sidebar')
   - Language selectors (elements with ID or class containing 'p-lang')
   - Table of contents (elements with ID or class containing 'toc')
   - Edit sections and jump links
   - Script and style tags

3. **Text Extraction**: The main article content was extracted from the `.mw-parser-output` div, which contains the core Wikipedia article text

4. **Text Cleaning**: The extracted text was cleaned to:
   - Remove excessive whitespace
   - Preserve paragraph structure
   - Maintain readability

The plaintext was saved to the `plaintext/` directory with filenames following the pattern `{docID}_{sanitized_topic}.txt`.

### PDF Generation

Two types of PDF files were generated from the HTML content:

#### Standard PDF (Text-Based)

Standard PDFs were created using WeasyPrint, a Python library for converting HTML/CSS to PDF. The process involved:

1. **HTML Cleaning**: The HTML was preprocessed to remove Wikipedia UI elements (same as plaintext extraction)

2. **CSS Styling**: Custom CSS was applied for better PDF rendering:
   - Page size set to A4 with 2cm margins
   - Font family set to Arial with appropriate line height
   - Wikipedia UI elements hidden using CSS display rules

3. **PDF Generation**: The cleaned HTML was converted to PDF format and saved to the `pdf/` directory

#### Image-Based PDF (for OCR Testing)

Image-based PDFs were created for potential OCR (Optical Character Recognition) testing scenarios. Two methods were implemented:

1. **Primary Method (Playwright)**: Using Playwright to render the HTML in a headless browser and capture screenshots, which were then compiled into a PDF. This method provides high-quality rendering that closely matches web browser display.

2. **Fallback Method (WeasyPrint + pdf2image)**: If Playwright fails, the standard PDF is converted to images using pdf2image and then recompiled into an image-based PDF.

These image-based PDFs were saved to the `pdf_images/` directory. They contain no text layer, making them suitable for testing OCR-based extraction methods.

### Document Formatting for RAG

The plaintext files were then processed to create a structured JSONL (JSON Lines) format file (`docs.jsonl`) suitable for RAG systems. Each line in the JSONL file contains a JSON object with the following structure:

```json
{
  "docID": 0,
  "source": "https://cs.wikipedia.org/wiki/...",
  "text": "title: {title}\nurl: {url}\ntext: {content}"
}
```

This format was created using the `create_docs_rag_format.py` script, which:
1. Reads the `links.csv` file to obtain docID, title, and URL mappings
2. Matches each entry with its corresponding plaintext file
3. Formats the text with metadata (title and URL) prepended to the content
4. Writes each document as a separate JSON object on a single line

## Question Generation

### Overview

Questions for evaluating the RAG system were generated using a large language model (LLM) via the EINFRA API. The question generation process was implemented in `generate_wikipedia_questions.py` (located in `embedders_experiments/`).

### Summary Generation

Before generating questions, a summary was created for each document using the LLM. The summary generation process:

1. **Input**: The first 8,000 characters of each document's text, along with the document title and URL

2. **Prompt**: A structured prompt instructing the model to extract:
   - Document name (název dokumentu)
   - Keywords (3-7 key terms)
   - One-sentence summary (shrnutí jednou větou)
   - Main topics (3-5 main topics or sections)
   - Important facts (3-5 specific facts, dates, names, or events)

3. **Output Format**: JSON structure containing the extracted information

4. **Model**: Meta-Llama-3.1-70B-Instruct via EINFRA API with temperature set to 0.3 for more deterministic outputs

### Question Generation Process

Questions were generated using the following methodology:

1. **Input Preparation**:
   - Document metadata (docID, source, title)
   - Generated summary (in JSON format)
   - Document excerpt (first 3,000 characters to manage token limits)

2. **Question Types**: The LLM was instructed to generate diverse question types:
   - **Direct questions** (přímé): Questions that can be answered directly from the document
   - **Indirect questions** (nepřímé): Questions requiring inference or synthesis
   - **Paraphrased questions** (parafráze): Reformulations of direct questions
   - **Trick questions** (záludné): Questions that test understanding and may have subtle answers
   - **Summary questions** (souhrnné): Questions about the overall content or main themes

3. **Prompt Structure**: The prompt included:
   - Instructions for creating retrieval evaluation questions
   - Requirement that questions be answerable only from the given document
   - Specification of the desired JSON output format
   - Examples of question types

4. **Output Format**: Each generated question includes:
   - Question text (question)
   - Answer text (answer)
   - Question type (type)
   - Expected document ID (docID)

5. **Model Parameters**: 
   - Model: Meta-Llama-3.1-70B-Instruct
   - Temperature: 0.8 (higher than summary generation for more diverse questions)
   - Max retries: 3 attempts with exponential backoff

6. **Rate Limiting**: A 1-second delay was implemented between API calls to respect rate limits

### Question Storage

Generated questions were saved in JSONL format (`queries_generated.jsonl`) with the following structure:

```json
{
  "queryID": 0,
  "query": "Question text",
  "docID": [0],
  "source": ["https://cs.wikipedia.org/wiki/..."],
  "label": [1.0],
  "type": "direct",
  "answer": "Answer text"
}
```

Each question is associated with:
- A unique queryID
- The question text
- The document ID(s) that contain the answer
- The source URL(s)
- A relevance label (1.0 for relevant documents)
- The question type
- The expected answer

## File Structure

The final dataset structure is as follows:

```
wikipedia/
├── links.csv                    # Manual URL list with docID, title, url
├── docs.jsonl                   # Structured documents for RAG
├── queries_generated.jsonl      # Generated questions and answers
├── html/                        # Raw HTML files
│   ├── 0_umělá_inteligence.html
│   └── ...
├── plaintext/                   # Extracted plaintext
│   ├── 0_umělá_inteligence.txt
│   └── ...
├── pdf/                         # Standard PDFs (text-based)
│   ├── 0_umělá_inteligence.pdf
│   └── ...
└── pdf_images/                  # Image-based PDFs (for OCR)
    ├── 0_umělá_inteligence.pdf
    └── ...
```

## Technical Implementation Details

### Dependencies

The scraping and processing scripts require the following Python libraries:
- `requests`: HTTP requests for downloading HTML
- `beautifulsoup4`: HTML parsing and content extraction
- `weasyprint`: HTML to PDF conversion
- `playwright`: Browser automation for image-based PDF generation
- `pdf2image`: PDF to image conversion (fallback method)
- `Pillow`: Image processing
- `tqdm`: Progress bars
- `python-dotenv`: Environment variable management (for API keys)

### Error Handling

The scripts include error handling for:
- Network failures during HTML download
- Missing or malformed HTML content
- PDF conversion failures
- API rate limiting and failures
- File I/O errors

### Reproducibility

The process is designed to be reproducible:
- All URLs are stored in CSV format
- File naming follows consistent patterns
- Scripts can be re-run with a `--force` flag to regenerate files
- Existing files are skipped by default to avoid redundant processing

## Limitations and Considerations

1. **Source Reliability**: Wikipedia content may contain inaccuracies or be subject to editing. Users should verify critical information.

2. **Language**: All content is in Czech, which may limit applicability to English-language RAG systems.

3. **Temporal Validity**: Wikipedia articles are subject to updates. The dataset represents a snapshot at the time of collection.

4. **Coverage**: The dataset covers 20 articles, which may not represent the full breadth of informatics and NLP topics.

5. **Question Quality**: Generated questions depend on LLM quality and may require manual review and filtering.

6. **Copyright**: Wikipedia content is available under Creative Commons Attribution-ShareAlike license, but users should verify current licensing terms.

## Conclusion

This dataset provides a structured collection of Czech Wikipedia articles on informatics and natural language processing, processed into multiple formats (HTML, plaintext, PDF) suitable for testing RAG systems. The manual curation process ensured relevance, while automated scraping and question generation enabled efficient dataset creation. The resulting dataset serves as a foundation for evaluating retrieval and generation capabilities in a controlled, reproducible manner.

