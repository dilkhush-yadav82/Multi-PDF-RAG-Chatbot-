import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# ---------------------
# Load ENV
# ---------------------
load_dotenv()

# ---------------------
# Embedding Model
# ---------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------
# LLM
# ---------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# ---------------------
# 1️⃣ Process PDFs
# ---------------------
def process_documents(uploaded_files):

    all_docs = []
    failed_files = []

    for file in uploaded_files:
        temp_path = f"temp_{file.name}"

        try:
            # Save file temporarily
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())

            # Load PDF
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            # If empty → mark failed
            if not docs:
                failed_files.append(file.name)
                continue

            # Add metadata
            for doc in docs:
                doc.metadata["source"] = file.name

            all_docs.extend(docs)

        except Exception:
            failed_files.append(file.name)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ❌ If no valid docs
    if not all_docs:
        return None, failed_files

    # ---------------------
    # Split
    # ---------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_docs)

    if not chunks:
        return None, failed_files

    # ---------------------
    # Vector DB
    # ---------------------
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )

    return vectordb, failed_files


# ---------------------
# 2️⃣ QA Function
# ---------------------
def answer_question(vectordb, query):

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain({"query": query})

    answer = response["result"]

    sources = list(set([
        doc.metadata.get("source", "Unknown")
        for doc in response["source_documents"]
    ]))

    return answer, sources
