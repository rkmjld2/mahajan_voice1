import streamlit as st
import fitz  # This is PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader  # Still use this wrapper if you prefer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from gtts import gTTS
import tempfile
import os
import io  # For BytesIO

st.title("📄 RAG PDF Voice Assistant (Groq)")

st.info("Upload a PDF (try your sample.pdf first!) → Ask questions → Get text + voice answer.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    # Read bytes once
    pdf_bytes = uploaded_file.read()

    # Quick size check
    if len(pdf_bytes) < 200:
        st.error("Uploaded file is empty or too small. Please upload a valid PDF.")
    else:
        try:
            with st.spinner("Loading PDF from memory (no disk write)..."):
                # Option 1: Preferred - direct stream loading with fitz (very reliable)
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                # Extract text manually if needed, but for LangChain we can wrap it
                # Option 2: Use PyMuPDFLoader with BytesIO (also good)
                # pdf_stream = io.BytesIO(pdf_bytes)
                # loader = PyMuPDFLoader(pdf_stream)  # Some versions support stream= directly
                # docs = loader.load()

                # If using manual fitz → convert to LangChain Documents
                from langchain_core.documents import Document
                docs = []
                for page_num in range(len(doc)):
                    text = doc[page_num].get_text("text")
                    metadata = {"source": uploaded_file.name, "page": page_num + 1}
                    docs.append(Document(page_content=text, metadata=metadata))
                
                doc.close()  # Important: close fitz doc

                if not docs or all(len(d.page_content.strip()) == 0 for d in docs):
                    st.warning("No readable text extracted.\nLikely image-only (scanned) PDF or format issue.")
                else:
                    st.success(f"PDF loaded successfully! {len(docs)} page(s) extracted.")

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

                    # LLM
                    llm = ChatGroq(
                        groq_api_key=st.secrets["GROQ_API_KEY"],
                        model_name="llama3-70b-8192",   # ← This is the culprit
                        temperature=0.3
                      )  
                    prompt = ChatPromptTemplate.from_template(
                        """Answer based only on the provided context. Be concise and accurate.

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

                    question = st.text_input("Ask a question about the PDF:")

                    if question and question.strip():
                        with st.spinner("Generating answer..."):
                            response = rag_chain.invoke(question)
                            answer_text = response.content.strip()

                            st.subheader("Answer")
                            st.markdown(answer_text)

                            # Voice
                            with st.spinner("Preparing audio..."):
                                tts = gTTS(text=answer_text, lang='en')
                                audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                                tts.save(audio_tmp.name)
                                audio_path = audio_tmp.name

                                st.audio(audio_path, format="audio/mp3")

                                with open(audio_path, "rb") as af:
                                    st.download_button(
                                        label="Download voice answer (MP3)",
                                        data=af,
                                        file_name="answer.mp3",
                                        mime="audio/mp3"
                                    )

        except Exception as e:
            st.error(f"PDF processing failed:\n\n{str(e)}\n\n"
                     "For your sample.pdf content → it should work fine.\n"
                     "If not: Check if file uploaded fully (size >0), no password, not corrupted.\n"
                     "Test: Save the text you shared as .pdf from Word/Notepad → upload that.")

        finally:
            # Cleanup audio if created
            if 'audio_path' in locals() and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass

