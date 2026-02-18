import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
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
        "Answer the question using the context:\n\n{context}\n\nQuestion: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    question = st.text_input("Ask a question")

    if question:
        answer = rag_chain.invoke(question)
        st.write(answer.content)

        tts = gTTS(answer.content)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)

        st.audio(tmp_file.name)
