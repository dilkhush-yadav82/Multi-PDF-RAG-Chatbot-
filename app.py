import streamlit as st
from rag_utility import process_documents, answer_question

st.set_page_config(page_title="RAG ChatBot",page_icon="🤖", layout="wide")
st.title("🦙 Multi-PDF RAG Chatbot")

# Store vectordb in session
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# Upload multiple PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# Process button
if st.button("Process Documents"):
    if uploaded_files:
        st.session_state.vectordb = process_documents(uploaded_files)
        st.success("Documents processed successfully ✅")
    else:
        st.warning("Please upload at least one PDF")

# Ask question
query = st.text_area("Ask a question")

if st.button("Get Answer"):
    if st.session_state.vectordb is None:
        st.warning("Please upload and process documents first")
    else:
        answer, sources = answer_question(
            st.session_state.vectordb,
            query
        )

        st.markdown(" 📌 Answer")
        st.write(answer)

        st.markdown(" 📄 Sources")
        for src in sources:
            st.write(f"- {src}")
