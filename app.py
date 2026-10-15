
import streamlit as st
import docx
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Document Summarizer | IR Project",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# 2. DOCUMENT TEXT EXTRACTION
# --------------------------------------------------

def extract_text(uploaded_file):
    """Extract paragraphs and table text from a DOCX file."""

    document = docx.Document(BytesIO(uploaded_file.getvalue()))

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    # Include text from tables in the Word document.
    for table in document.tables:
        for row in table.rows:
            row_text = " ".join(
                cell.text.strip() for cell in row.cells
                if cell.text.strip()
            )
            if row_text:
                paragraphs.append(row_text)

    return "\n".join(paragraphs)


# --------------------------------------------------
# 3. SENTENCE SEGMENTATION
# --------------------------------------------------

def split_sentences(text):
    """Split document text into sentences."""

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Simple sentence boundary detection.
    sentences = re.split(r'(?<=[.!?])\s+', text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# --------------------------------------------------
# 4. TF-IDF BASED SUMMARIZATION
# --------------------------------------------------

def summarize_document(text, summary_ratio=0.25):
    """
    Generate an extractive summary using TF-IDF and
    cosine similarity to the document's TF-IDF centroid.
    """

    sentences = split_sentences(text)

    if not sentences:
        return [], sentences, []

    if len(sentences) <= 2:
        return sentences, sentences, [1.0] * len(sentences)

    # Convert sentences into TF-IDF vectors.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Handles documents with no usable vocabulary.
        return sentences, sentences, [0.0] * len(sentences)

    # Represent the overall document using its
    # average sentence TF-IDF vector.
    document_vector = np.asarray(
        tfidf_matrix.mean(axis=0)
    )

    # Calculate similarity between every sentence
    # and the overall document representation.
    scores = cosine_similarity(
        tfidf_matrix,
        document_vector
    ).flatten()

    # Determine how many sentences to retain.
    summary_count = max(
        1,
        int(np.ceil(len(sentences) * summary_ratio))
    )

    summary_count = min(summary_count, len(sentences))

    # Select the highest-ranked sentences.
    ranked_indices = np.argsort(scores)[::-1][:summary_count]

    # Restore original sentence order for readability.
    selected_indices = sorted(ranked_indices)

    summary = [sentences[i] for i in selected_indices]

    return summary, sentences, scores.tolist()


# --------------------------------------------------
# 5. USER INTERFACE
# --------------------------------------------------

st.title("📄 Intelligent Document Summarization")
st.caption(
    "Information Retrieval Mini Project | "
    "TF-IDF + Cosine Similarity"
)

st.markdown(
    """
    Upload a **Microsoft Word (.docx)** document to extract
    important sentences and generate a concise summary.
    """
)

uploaded_file = st.file_uploader(
    "Upload your Word document",
    type=["docx"]
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


# --------------------------------------------------
# 6. PROCESS DOCUMENT
# --------------------------------------------------

if uploaded_file is not None:

    try:
        text = extract_text(uploaded_file)

        if not text.strip():
            st.error(
                "No readable text was found in this document."
            )
            st.stop()

        summary, sentences, scores = summarize_document(
            text,
            summary_ratio
        )

        if not sentences:
            st.error("Unable to identify sentences in the document.")
            st.stop()

        # Summary statistics.
        original_words = len(text.split())
        summary_text = " ".join(summary)
        summary_words = len(summary_text.split())

        compression = (
            (1 - summary_words / original_words) * 100
            if original_words else 0
        )

        st.success("Document processed successfully!")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Words", original_words)
        col2.metric("Total Sentences", len(sentences))
        col3.metric("Summary Words", summary_words)
        col4.metric("Compression", f"{compression:.1f}%")

        st.divider()

        # Display generated summary.
        st.subheader("📝 Generated Summary")

        st.text_area(
            "Extractive summary",
            value=summary_text,
            height=250
        )

        # Download summary as a text file.
        st.download_button(
            label="⬇️ Download Summary",
            data=summary_text.encode("utf-8"),
            file_name="document_summary.txt",
            mime="text/plain"
        )

        # Display extracted original text.
        with st.expander("View Original Extracted Text"):
            st.write(text)

        # Display sentence rankings.
        if show_scores:
            st.subheader("📊 Sentence Importance Ranking")

            ranked = sorted(
                enumerate(scores),
                key=lambda item: item[1],
                reverse=True
            )

            for rank, (index, score) in enumerate(ranked, start=1):
                st.markdown(
                    f"**Rank {rank}** | "
                    f"Sentence {index + 1} | "
                    f"Score: `{score:.4f}`"
                )
                st.write(sentences[index])

                st.progress(
                    max(0.0, min(1.0, float(score)))
                )

    except Exception as error:
        st.error(
            "An error occurred while processing the document. "
            "Please check that it is a valid, non-corrupted DOCX file."
        )
        st.exception(error)

else:
    st.info("Upload a .docx file to get started.")

    st.markdown(
        """
        ### How it works
        1. Extract text from the Word document.
        2. Divide the text into sentences.
        3. Calculate TF-IDF vectors for each sentence.
        4. Calculate cosine similarity against the document vector.
        5. Select the top-ranked sentences.
        6. Arrange them in their original order and generate a summary.
        """
    )
