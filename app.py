
import streamlit as st
import docx
import fitz  # PyMuPDF
import re
import numpy as np

from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Research Paper Analyzer | IR Project",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# 2. DOCUMENT TEXT EXTRACTION
# --------------------------------------------------

def extract_docx_text(uploaded_file):
    """Extract paragraphs and table text from DOCX."""

    document = docx.Document(
        BytesIO(uploaded_file.getvalue())
    )

    parts = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    for table in document.tables:
        for row in table.rows:
            cells = [
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            ]

            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def extract_pdf_text(uploaded_file):
    """Extract text from every page of a PDF."""

    pdf_bytes = uploaded_file.getvalue()
    pages = []

    with fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:

        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()

            if page_text:
                pages.append(
                    f"\n{page_text}"
                )

    return "\n".join(pages).strip()


def extract_document_text(uploaded_file):
    """Choose the appropriate extraction method."""

    extension = uploaded_file.name.lower()

    if extension.endswith(".pdf"):
        return extract_pdf_text(uploaded_file)

    if extension.endswith(".docx"):
        return extract_docx_text(uploaded_file)

    raise ValueError("Unsupported document format.")


# --------------------------------------------------
# 3. SENTENCE SEGMENTATION
# --------------------------------------------------

def split_sentences(text):
    """Split extracted text into sentences."""

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Avoid splitting common academic abbreviations.
    text = re.sub(
        r"\b(e\.g|i\.e|et al|Fig|Eq|Dr|Prof|vs|No)\.",
        lambda match: match.group(0).replace(".", "<DOT>"),
        text,
        flags=re.IGNORECASE
    )

    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9"\[])',
        text
    )

    return [
        sentence.replace("<DOT>", ".").strip()
        for sentence in sentences
        if sentence.strip()
    ]


# --------------------------------------------------
# 4. TF-IDF SUMMARIZATION
# --------------------------------------------------

def summarize_document(text, summary_ratio=0.25):
    """
    Extractive summarization using TF-IDF sentence vectors
    and cosine similarity against the document centroid.
    """

    sentences = split_sentences(text)

    if not sentences:
        return [], sentences, []

    if len(sentences) <= 2:
        return sentences, sentences, [1.0] * len(sentences)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)

    except ValueError:
        return sentences, sentences, [0.0] * len(sentences)

    document_vector = np.asarray(
        tfidf_matrix.mean(axis=0)
    )

    scores = cosine_similarity(
        tfidf_matrix,
        document_vector
    ).flatten()

    summary_count = max(
        1,
        int(np.ceil(len(sentences) * summary_ratio))
    )

    summary_count = min(
        summary_count,
        len(sentences)
    )

    ranked_indices = np.argsort(scores)[::-1][
        :summary_count
    ]

    # Preserve the original document order.
    selected_indices = sorted(ranked_indices)

    summary = [
        sentences[index]
        for index in selected_indices
    ]

    return summary, sentences, scores.tolist()


# --------------------------------------------------
# 5. RESEARCH PAPER SECTION DETECTION
# --------------------------------------------------

SECTION_ALIASES = {
    "Abstract": [
        "abstract"
    ],
    "Introduction": [
        "introduction",
        "background",
        "1 introduction",
        "1. introduction"
    ],
    "Related Work": [
        "related work",
        "literature review",
        "related studies",
        "previous work",
        "state of the art"
    ],
    "Methodology": [
        "methodology",
        "methods",
        "materials and methods",
        "proposed method",
        "proposed methodology",
        "method"
    ],
    "Results": [
        "results",
        "experimental results",
        "evaluation",
        "results and analysis"
    ],
    "Discussion": [
        "discussion",
        "results and discussion"
    ],
    "Conclusion": [
        "conclusion",
        "conclusions",
        "conclusion and future work",
        "conclusions and future work",
        "future work"
    ],
    "References": [
        "references",
        "bibliography"
    ]
}


def normalize_heading(line):
    """Normalize a potential section heading."""

    line = line.strip().lower()

    # Remove common numbering: 1, 1.2, 2.3.1, etc.
    line = re.sub(
        r"^\d+(?:\.\d+)*\.?\s*",
        "",
        line
    )

    line = re.sub(r"[:.\s]+$", "", line)

    return line.strip()


def detect_sections(text):
    """
    Detect common academic sections using headings
    on separate lines. Detection is heuristic.
    """

    lines = text.splitlines()
    detected = []

    for index, line in enumerate(lines):
        normalized = normalize_heading(line)

        if not normalized:
            continue

        for section_name, aliases in SECTION_ALIASES.items():
            if normalized in aliases:
                detected.append(
                    (index, section_name)
                )
                break

    # Remove repeated headings of the same type only
    # when they occur at the same position.
    sections = {}

    for position, name in detected:
        sections.setdefault(name, []).append(position)

    if not detected:
        return {}

    # Use the first occurrence of each recognized section.
    # Ignore references and everything following them.
    unique = []
    seen = set()

    for position, name in detected:
        if name in seen:
            continue

        unique.append((position, name))
        seen.add(name)

    unique.sort(key=lambda item: item[0])

    result = {}

    for i, (start, name) in enumerate(unique):
        if name == "References":
            break

        end = (
            unique[i + 1][0]
            if i + 1 < len(unique)
            else len(lines)
        )

        section_text = "\n".join(
            lines[start + 1:end]
        ).strip()

        if section_text:
            result[name] = section_text

    return result


# --------------------------------------------------
# 6. PAPER ANALYSIS HELPERS
# --------------------------------------------------

def analyze_paper(text, summary_ratio):
    """Generate overall and section-wise summaries."""

    overall_summary, sentences, scores = summarize_document(
        text,
        summary_ratio
    )

    sections = detect_sections(text)

    section_summaries = {}

    for section_name, section_text in sections.items():
        if section_name == "Abstract":
            # The abstract is already a concise author-written
            # summary, so display it without further compression.
            section_summaries[section_name] = section_text
        else:
            summary, _, _ = summarize_document(
                section_text,
                summary_ratio
            )

            if summary:
                section_summaries[section_name] = " ".join(
                    summary
                )

    return {
        "summary": overall_summary,
        "sentences": sentences,
        "scores": scores,
        "sections": sections,
        "section_summaries": section_summaries
    }


# --------------------------------------------------
# 7. USER INTERFACE
# --------------------------------------------------

st.title("📄 Intelligent Document & Research Paper Analyzer")

st.caption(
    "Information Retrieval Mini Project | "
    "TF-IDF + Cosine Similarity"
)

st.markdown(
    """
    Upload a **PDF research paper** or **Word document** to
    extract text, identify important academic sections,
    generate an extractive summary, and rank sentences
    according to their importance.
    """
)

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx"],
    help="Supports text-based PDFs and DOCX documents."
)

st.sidebar.header("Summarization Settings")

summary_ratio = st.sidebar.slider(
    "Summary length (% of sentences)",
    min_value=10,
    max_value=70,
    value=25,
    step=5
) / 100

show_scores = st.sidebar.checkbox(
    "Show sentence importance scores",
    value=True
)

show_sections = st.sidebar.checkbox(
    "Show section-wise paper analysis",
    value=True
)


# --------------------------------------------------
# 8. PROCESS DOCUMENT
# --------------------------------------------------

if uploaded_file is not None:

    try:
        with st.spinner(
            "Extracting text and analyzing the document..."
        ):
            text = extract_document_text(uploaded_file)

        if not text.strip():
            st.error(
                "No readable text was extracted. "
                "This PDF may be scanned or image-only. "
                "Run OCR before analyzing scanned PDFs."
            )
            st.stop()

        analysis = analyze_paper(
            text,
            summary_ratio
        )

        summary = analysis["summary"]
        sentences = analysis["sentences"]
        scores = analysis["scores"]
        sections = analysis["sections"]
        section_summaries = analysis["section_summaries"]

        if not sentences:
            st.error(
                "Unable to identify sentences in the document."
            )
            st.stop()

        # ------------------------------------------
        # DOCUMENT STATISTICS
        # ------------------------------------------

        original_words = len(text.split())
        summary_text = " ".join(summary)
        summary_words = len(summary_text.split())

        compression = (
            (1 - summary_words / original_words) * 100
            if original_words
            else 0
        )

        st.success("Document analyzed successfully!")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Words",
            f"{original_words:,}"
        )

        col2.metric(
            "Total Sentences",
            f"{len(sentences):,}"
        )

        col3.metric(
            "Summary Words",
            f"{summary_words:,}"
        )

        col4.metric(
            "Compression",
            f"{compression:.1f}%"
        )

        st.divider()

        # ------------------------------------------
        # OVERALL SUMMARY
        # ------------------------------------------

        st.subheader("📝 Overall Document Summary")

        st.text_area(
            "Extractive summary",
            value=summary_text,
            height=250
        )

        st.download_button(
            label="⬇️ Download Overall Summary",
            data=summary_text.encode("utf-8"),
            file_name="document_summary.txt",
            mime="text/plain"
        )

        # ------------------------------------------
        # RESEARCH PAPER ANALYSIS
        # ------------------------------------------

        if show_sections:
            st.divider()
            st.subheader("🔬 Research Paper Analysis")

            if sections:
                st.caption(
                    "Sections are detected from recognizable "
                    "headings in the extracted text. "
                    "Check the source paper if headings appear missing."
                )

                st.write(
                    f"**Sections identified:** "
                    f"{len(sections)}"
                )

                tabs = st.tabs(
                    list(section_summaries.keys())
                )

                for tab, (name, section_summary) in zip(
                    tabs,
                    section_summaries.items()
                ):
                    with tab:
                        st.markdown(f"### {name}")

                        st.write(section_summary)

                        section_download = (
                            f"{name}\n\n{section_summary}"
                        )

                        st.download_button(
                            label=f"Download {name} Summary",
                            data=section_download.encode("utf-8"),
                            file_name=(
                                re.sub(
                                    r"[^a-zA-Z0-9_-]",
                                    "_",
                                    name.lower()
                                )
                                + "_summary.txt"
                            ),
                            mime="text/plain",
                            key=f"download_{name}"
                        )

            else:
                st.info(
                    "No standard research paper headings were "
                    "recognized. The overall summary is still "
                    "available. Some papers use non-standard "
                    "headings or PDF layouts."
                )

        # ------------------------------------------
        # ORIGINAL TEXT
        # ------------------------------------------

        with st.expander(
            "📖 View Original Extracted Text"
        ):
            st.text_area(
                "Extracted document text",
                value=text,
                height=350
            )

        # ------------------------------------------
        # SENTENCE RANKINGS
        # ------------------------------------------

        if show_scores:
            st.divider()
            st.subheader("📊 Sentence Importance Ranking")

            ranked = sorted(
                enumerate(scores),
                key=lambda item: item[1],
                reverse=True
            )

            for rank, (index, score) in enumerate(
                ranked,
                start=1
            ):
                st.markdown(
                    f"**Rank {rank}** | "
                    f"Sentence {index + 1} | "
                    f"Score: `{score:.4f}`"
                )

                st.write(sentences[index])

                st.progress(
                    max(
                        0.0,
                        min(1.0, float(score))
                    )
                )

        # ------------------------------------------
        # DOWNLOAD ALL ANALYSIS
        # ------------------------------------------

        st.divider()
        st.subheader("📦 Export Analysis")

        report_parts = [
            "DOCUMENT ANALYSIS REPORT",
            "=" * 50,
            "",
            "OVERALL SUMMARY",
            "-" * 30,
            summary_text,
            ""
        ]

        for name, section_summary in section_summaries.items():
            report_parts.extend([
                "",
                name.upper(),
                "-" * 30,
                section_summary
            ])

        report_text = "\n".join(report_parts)

        st.download_button(
            label="⬇️ Download Complete Analysis",
            data=report_text.encode("utf-8"),
            file_name="research_paper_analysis.txt",
            mime="text/plain"
        )

    except Exception as error:
        st.error(
            "An error occurred while processing the document. "
            "Please check that the file is valid and not corrupted."
        )

        st.exception(error)

else:
    st.info(
        "Upload a PDF research paper or DOCX document to get started."
    )

    st.markdown(
        """
        ### How the system works

        1. **Text extraction:** Read PDF pages or Word paragraphs and tables.
        2. **Sentence segmentation:** Divide the extracted text into sentences.
        3. **TF-IDF representation:** Convert sentences into weighted term vectors.
        4. **Cosine similarity:** Compare each sentence with the document centroid.
        5. **Sentence ranking:** Select the highest-scoring sentences.
        6. **Section detection:** Identify common research paper headings.
        7. **Section-wise summarization:** Summarize the recognized sections.
        8. **Export:** Download the overall summary or complete analysis.

        ### Limitations

        - Scanned PDFs require OCR.
        - Section detection depends on how headings are extracted.
        - TF-IDF scores lexical importance, not scientific correctness.
        - The method does not independently verify claims, equations,
          citations, or experimental results.
        """
    )
