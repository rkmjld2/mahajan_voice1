import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from gtts import gTTS
import tempfile

st.title("📄 RAG PDF Voice Assistant (Groq)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("sample.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("sample.pdf")
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
        "Answer the question based only on the context:\n\n{context}\n\nQuestion: {input}"
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    question = st.text_input("Ask a question")

    if question:
        response = qa_chain.invoke({"input": question})
        answer = response["answer"]

        st.write(answer)

        # 🔊 Text to Speech
        tts = gTTS(answer)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)

        st.audio(tmp_file.name)
