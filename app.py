import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader  # ← switched to this (more robust)
# from langchain_community.document_loaders import PyPDFLoader   # comment out or remove old one
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from gtts import gTTS
import tempfile
import os

st.title("📄 RAG PDF Voice Assistant (Groq)")

st.info("Upload a PDF → ask questions → get text + voice answer.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    # Save to temp file safely
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.getvalue())
        pdf_path = tmp_pdf.name

    # Basic size check
    if os.path.getsize(pdf_path) < 200:
        st.error("Uploaded file is empty or too small to be a valid PDF. Try another file.")
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
    else:
        try:
            with st.spinner("Loading PDF with PyMuPDF (robust parser)..."):
                loader = PyMuPDFLoader(pdf_path)  # No strict arg needed — very tolerant
                docs = loader.load()

                if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
                    st.warning("No extractable text found.\n"
                               "Likely causes:\n"
                               "• Scanned/image-only PDF (needs OCR)\n"
                               "• Encrypted/protected PDF\n"
                               "• File is corrupted")
                else:
                    # Splitting
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=120
                    )
                    splits = text_splitter.split_documents(docs)

                    # Embeddings & Vectorstore
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vectorstore = FAISS.from_documents(splits, embeddings)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                    # LLM setup
                    llm = ChatGroq(
                        groq_api_key=st.secrets["GROQ_API_KEY"],
                        model_name="llama3-70b-8192",
                        temperature=0.3
                    )

                    # Improved prompt
                    prompt = ChatPromptTemplate.from_template(
                        """Answer the question based only on the provided context. 
                        Be concise, accurate, and helpful.

                        Context:
                        {context}

                        Question: {question}

                        Answer:"""
                    )

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                    )

                    # Question input
                    question = st.text_input("Ask something about the PDF content:")

                    if question and question.strip():
                        with st.spinner("Generating answer..."):
                            response = rag_chain.invoke(question)
                            answer_text = response.content.strip()

                            st.subheader("Answer")
                            st.markdown(answer_text)

                            # Voice
                            with st.spinner("Creating audio..."):
                                tts = gTTS(text=answer_text, lang='en')
                                audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                                tts.save(audio_tmp.name)

                                st.audio(audio_tmp.name, format="audio/mp3")

                                # Download option
                                with open(audio_tmp.name, "rb") as af:
                                    st.download_button(
                                        "⬇️ Download spoken answer",
                                        af,
                                        file_name="answer.mp3",
                                        mime="audio/mp3"
                                    )

        except Exception as e:
            st.error(f"PDF processing failed:\n\n{str(e)}\n\n"
                     "Suggestions:\n"
                     "• Re-download/save the PDF from original source\n"
                     "• Test with a simple text PDF (e.g. create one in Word/Google Docs)\n"
                     "• If it's scanned → use an OCR tool first (like Adobe, online2pdf, etc.)\n"
                     "• File might be password-protected — remove protection if possible")

        finally:
            # Cleanup
            if os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                except:
                    pass
