import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate          # fixed
from langchain_core.runnables import RunnablePassthrough      # fixed
from gtts import gTTS
import tempfile
import os

st.title("📄 RAG PDF Voice Assistant (Groq)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        pdf_path = tmp_pdf.name

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        splits = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        llm = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"],
            model_name="llama3-70b-8192"
        )

        prompt = ChatPromptTemplate.from_template(
            "Answer the question using the context:\n\n{context}\n\nQuestion: {question}"
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        question = st.text_input("Ask a question about the PDF")

        if question:
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke(question)
                st.write("**Answer:**")
                st.write(answer.content)

                # Text-to-speech
                tts = gTTS(answer.content, lang='en')
                tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(tmp_audio.name)

                st.audio(tmp_audio.name)

    finally:
        # Clean up temporary PDF
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
