"""Minimal Streamlit UI. Run: PYTHONPATH=src streamlit run src/biorag/app.py

Serves the off-the-shelf retriever, not the fine-tuned one: the ablation showed
fine-tuning lowered MRR, so shipping it would knowingly serve the worse system.
"""

from pathlib import Path

import streamlit as st

from biorag.generate.answer import answer_question
from biorag.index.dense import load_index
from biorag.ingest.build_corpus import load_corpus
from biorag.retrieve.dense import DenseRetriever

INDEX_PATH = Path("data/base.faiss")


@st.cache_resource
def get_retriever() -> DenseRetriever:
    """Cached so the model and index load once, not on every keystroke."""
    chunks = load_corpus()
    return DenseRetriever(chunks, index=load_index(INDEX_PATH))


st.title("BioRAG")
st.caption("Question answering over musculoskeletal biomechanics abstracts")

question = st.text_input(
    "Question", placeholder="How does lifting posture affect spinal load?"
)

if question:
    with st.spinner("Retrieving and generating…"):
        result = answer_question(question, get_retriever())

    st.markdown(result.text)
    st.divider()
    st.subheader("Sources")
    for chunk in result.chunks:
        with st.expander(f"[{chunk.pmid}] {chunk.title}"):
            st.write(chunk.text)
            st.caption(f"{chunk.journal}, {chunk.year}")
