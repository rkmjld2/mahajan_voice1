import streamlit as st
import fitz  # PyMuPDF
import speech_recognition as sr
import pyttsx3
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama  # or Groq if configured


# --- Load PDF ---
def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def build_vectorstore(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak your question...")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        st.success(f"You asked: {query}")
        return query
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")
        return None

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

st.title("📚 Voice-powered RAG App")

pdf_file = "sample.pdf"
if pdf_file:
    text = load_pdf(pdf_file)
    vectorstore = build_vectorstore(text)
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama2")  # or Groq if configured
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if st.button("Ask by Voice"):
        query = get_voice_input()
        if query:
            answer = qa.run(query)
            st.write("Answer:", answer)
            speak_text(answer)

