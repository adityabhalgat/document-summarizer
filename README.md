# Intelligent Document & Research Paper Analyzer

An Information Retrieval mini-project that extracts text from research papers,
generates extractive summaries, detects common academic sections, and ranks
sentences by importance using **TF-IDF** and **cosine similarity**.

The project provides a browser-based interface built with
[Streamlit](https://streamlit.io/). Users can upload a text-based PDF or a
Microsoft Word (`.docx`) document and download the generated summaries and
analysis report.

## Features

- Extracts text from every page of a PDF.
- Extracts paragraphs and table content from DOCX files.
- Splits extracted content into sentences.
- Represents sentences using TF-IDF with unigram and bigram features.
- Scores sentences by cosine similarity to the document centroid.
- Produces an extractive summary while preserving the original sentence order.
- Detects common research-paper sections, including:
  - Abstract
  - Introduction
  - Related Work
  - Methodology
  - Results
  - Discussion
  - Conclusion
  - References
- Displays document statistics such as word count, sentence count, summary
  word count, and compression percentage.
- Allows users to adjust the summary length from 10% to 70%.
- Provides optional sentence scores and section-wise analysis.
- Downloads the overall summary, individual section summaries, or a complete
  analysis report as text files.

## Technology Stack

- **Python 3**
- **Streamlit** - web application interface
- **PyMuPDF** - PDF text extraction
- **python-docx** - DOCX text extraction
- **NumPy** - numerical operations
- **scikit-learn** - TF-IDF vectorization and cosine similarity

## Project Structure

```text
Mini-Project-IR/
├── app.py          # Streamlit application
├── requirements.txt # Python dependencies
├── India.docx      # Sample DOCX input document
└── README.md       # Project documentation
```

## Prerequisites

- Python 3.9 or newer
- `pip`
- A modern web browser

Python 3.13 has been used successfully with the included dependencies. A
virtual environment is recommended so that project packages do not affect
other Python projects.

## Installation

### macOS and Linux

Open a terminal in the project directory:

```bash
cd /path/to/Mini-Project-IR
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
cd path\to\Mini-Project-IR
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the following command in a PowerShell
window with the appropriate user permissions and then activate the
environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Running the Application

After activating the virtual environment, run:

```bash
streamlit run app.py
```

Streamlit prints a local URL in the terminal, normally:

```text
http://localhost:8501
```

Open that URL in a browser. To stop the application, press `Ctrl+C` in the
terminal.

If the `streamlit` command is not found, use the Python module form:

```bash
python -m streamlit run app.py
```

## Using the Application

1. Start the application with `streamlit run app.py`.
2. Select **Browse files** under **Upload your document**.
3. Upload a text-based `.pdf` or `.docx` file.
4. Choose the required **Summary length** in the sidebar.
5. Enable or disable:
   - **Show sentence importance scores**
   - **Show section-wise paper analysis**
6. Review the document statistics and overall summary.
7. Download any required output:
   - `document_summary.txt` - overall extractive summary
   - `<section>_summary.txt` - summary for a detected section
   - `research_paper_analysis.txt` - combined analysis report

The included `India.docx` file can be used as a sample input:

```text
India.docx -> Browse files -> Open
```

## How the Analysis Works

### 1. Text extraction

The application chooses an extractor based on the uploaded file extension:

- PDF files are read page by page using PyMuPDF.
- DOCX files are read from paragraphs and tables using `python-docx`.

### 2. Sentence segmentation

Whitespace is normalized and the text is split at likely sentence-ending
punctuation. Several common academic abbreviations, such as `e.g.`, `i.e.`,
`et al.`, `Fig.`, and `Eq.`, are protected to reduce incorrect splits.

### 3. TF-IDF representation

Each sentence is converted into a TF-IDF vector. The application uses English
stop-word removal, unigrams and bigrams, and sublinear term frequency.

### 4. Sentence scoring

The centroid of all sentence vectors represents the document. Each sentence is
scored by its cosine similarity to that centroid. Sentences with higher scores
are considered more representative of the document.

### 5. Summary generation

The configured summary percentage determines how many sentences are selected.
The highest-scoring sentences are selected, then returned in their original
document order so that the summary remains readable.

### 6. Section detection

Recognizable headings are detected when they appear on separate lines. Common
numbering such as `1`, `1.1`, or `2.3.1` is removed before matching heading
aliases. References and content after the first References heading are excluded
from section summaries.

## Output Metrics

- **Total Words:** Number of whitespace-separated words in the extracted text.
- **Total Sentences:** Number of sentences identified by the segmentation
  step.
- **Summary Words:** Number of words in the generated overall summary.
- **Compression:** Percentage reduction from the original text to the summary.

## Limitations

- Scanned or image-only PDFs do not contain selectable text and require OCR
  before uploading.
- Section detection is heuristic and depends on headings being extracted as
  separate lines.
- PDF layout, columns, headers, footers, and unusual formatting can affect
  extracted text.
- TF-IDF measures lexical similarity and representativeness; it does not
  evaluate scientific correctness, factual accuracy, or writing quality.
- The summarizer does not independently verify claims, equations, citations, or
  experimental results.
- The application accepts PDF and DOCX files only.

## Troubleshooting

### `ModuleNotFoundError`

Confirm that the virtual environment is active and reinstall the dependencies:

```bash
pip install -r requirements.txt
```

### The browser page does not open automatically

Copy the local URL shown in the terminal and open it manually in a browser.

### No readable text was extracted

The uploaded PDF may be scanned or image-only. Apply OCR to the document and
upload the searchable PDF again.

### No sections were detected

The document may use non-standard headings, or the PDF layout may not preserve
headings as separate lines. The overall summary remains available.

### The upload is rejected

Check that the file has a `.pdf` or `.docx` extension and is not corrupted.

## Academic Context

This project demonstrates a classical extractive information-retrieval
pipeline. It does not use a generative language model. The generated summary
is composed of sentences selected from the original document according to
TF-IDF similarity scores.