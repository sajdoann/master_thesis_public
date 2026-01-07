# PDF Text Extraction

### Create wikipedia dataset
- input: urls csv list of 20 wikipeida articles
- download html
- delete unuseful information (fe all supported languages list, headers, footers) to get the text about the topic
- make plaintext
- make pdfs (pdf with text layer, pdf image)

pdf images:
pdf2image.convert_from_path() to convert each PDF page to a PNG image
Resolution: 300 DPI
Returns a list of PIL Image objects (one per page)

### Compare extractors
- difflib for sequence comparision score 0-1