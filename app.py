import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
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

st.info("Upload a PDF file and ask questions about its content. Voice answer included.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.getvalue())  # safer than read() in some cases
        pdf_path = tmp_pdf.name

    # Quick check if file has content
    if os.path.getsize(pdf_path) < 100:
        st.error("The uploaded file appears to be empty or too small. Please upload a valid PDF.")
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
    else:
        try:
            with st.spinner("Loading and processing PDF... (may take a few seconds)"):
                # Load PDF - strict=False helps with many imperfect PDFs
                loader = PyPDFLoader(pdf_path, strict=False)
                docs = loader.load()

                if not docs or len(docs) == 0:
                    st.warning("No readable text could be extracted from this PDF.\n"
                               "Possible reasons: scanned image-only PDF, encrypted file, or severe corruption.")
                else:
                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=120,
                        length_function=len,
                    )
                    splits = text_splitter.split_documents(docs)

                    # Create embeddings and vector store
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vectorstore = FAISS.from_documents(splits, embeddings)

                    # Retriever
                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )

                    # LLM
                    llm = ChatGroq(
                        groq_api_key=st.secrets["GROQ_API_KEY"],
                        model_name="llama3-70b-8192",
                        temperature=0.3,
                    )

                    # Prompt
                    prompt = ChatPromptTemplate.from_template(
                        """You are a helpful assistant that answers questions based only on the provided context. 
                        Be concise, accurate and polite.

                        Context:
                        {context}

                        Question: {question}

                        Answer:"""
                    )

                    # Format retrieved documents
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    # RAG chain
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                    )

                    # User question
                    question = st.text_input("Ask a question about the document:", key="question_input")

                    if question.strip():
                        with st.spinner("Generating answer..."):
                            try:
                                response = rag_chain.invoke(question)
                                answer_text = response.content

                                st.subheader("Answer:")
                                st.markdown(answer_text)

                                # Text-to-Speech
                                with st.spinner("Preparing voice..."):
                                    tts = gTTS(text=answer_text, lang='en', slow=False)
                                    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                                    tts.save(audio_file.name)

                                    st.audio(audio_file.name, format="audio/mp3")

                                    # Optional: offer download
                                    with open(audio_file.name, "rb") as f:
                                        st.download_button(
                                            label="Download voice answer (MP3)",
                                            data=f,
                                            file_name="answer.mp3",
                                            mime="audio/mp3"
                                        )

                            except Exception as inner_e:
                                st.error(f"Error while generating answer:\n{str(inner_e)}")

        except Exception as e:
            st.error(f"Could not process the PDF file.\n\nError: {str(e)}\n\n"
                     "Common causes:\n"
                     "• File is corrupted or incomplete\n"
                     "• PDF is password protected\n"
                     "• PDF contains only scanned images (no text)\n"
                     "• File was not fully uploaded\n\n"
                     "Try: re-save/export the PDF or use a different file.")

        finally:
            # Always try to clean up
            if os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                except:
                    pass
