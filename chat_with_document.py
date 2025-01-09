import streamlit as st
import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import os


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''  # Handle pages with no text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


# Function to create a vector store from text
def create_vector_store(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


# Function to build a QA chain
def build_qa_chain(vector_store):
    try:
        qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        llm = HuggingFacePipeline(pipeline=qa_model)

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the following context to answer the question:\n\n"
                "{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": qa_prompt},
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error building QA chain: {e}")
        return None


# Streamlit App Interface
st.title("PDF Chat Application")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with st.spinner("Processing the PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            # Handle large files by limiting text size
            if len(pdf_text) > 1_000_000:  # Limit to ~1 million characters
                st.warning("The uploaded file is too large. Only the first part of the document will be processed.")
                pdf_text = pdf_text[:1_000_000]

            vector_store = create_vector_store(pdf_text)
            if vector_store:
                qa_chain = build_qa_chain(vector_store)
                if qa_chain:
                    query = st.text_input("Ask a question about the document:")
                    if query:
                        try:
                            with st.spinner("Finding the answer..."):
                                retrieved_context = vector_store.similarity_search(query, k=1)
                                if retrieved_context:
                                    st.write("**Retrieved Context:**", retrieved_context[0].page_content)
                                else:
                                    st.warning("No relevant context found in the document.")

                                response = qa_chain.run(query)
                                if response.strip():
                                    st.write("**Answer:**", response)
                                else:
                                    st.write("I'm sorry, I couldn't find an answer to your question.")
                        except Exception as e:
                            st.error(f"Error during QA: {e}")

# Error handling for no file uploaded
if not uploaded_file:
    st.info("Please upload a PDF file to get started.")
