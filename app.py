import streamlit as st
from rag_utility import process_documents, answer_question

st.set_page_config(page_title="RAG ChatBot",page_icon="🤖", layout="wide")
st.title("📄 Multi-PDF RAG Chatbot")

# Upload multiple PDFs
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Process PDFs
if uploaded_files:
    vectordb, failed_files = process_documents(uploaded_files)

    if vectordb is None:
        st.error("❌ No valid text found in uploaded PDFs.")
    else:
        st.session_state.vectordb = vectordb
        st.success("✅ Documents processed successfully!")

    if failed_files:
        st.warning(f"⚠️ Failed to process: {', '.join(failed_files)}")

# Ask question
query = st.text_input("Ask a question")

if st.button("Get Answer"):

    if "vectordb" not in st.session_state:
        st.error("⚠️ Please upload and process PDFs first.")
    else:
        answer, sources = answer_question(st.session_state.vectordb, query)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for src in sources:
            st.write(f"- {src}")
