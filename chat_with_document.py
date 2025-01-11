import streamlit as st
import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import pipeline
import re
from datetime import datetime

# Function to extract experience durations and calculate total work experience
def calculate_total_experience(text):
    try:
        # Regex to extract work experience durations in "Month Year – Month Year" format
        date_ranges = re.findall(
            r"(\w{3,9} \d{4})\s*–\s*(\w{3,9} \d{4}|Present)",
            text,
        )

        total_months = 0
        for start, end in date_ranges:
            # Convert "Present" to the current date
            end_date = datetime.now() if end.lower() == "present" else datetime.strptime(end, "%b %Y")
            start_date = datetime.strptime(start, "%b %Y")

            # Calculate the duration in months
            total_months += (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        # Convert total months to years and months
        years = total_months // 12
        months = total_months % 12
        return years, months
    except Exception as e:
        st.error(f"Error calculating total experience: {e}")
        return 0, 0

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

            # Calculate total work experience
            years, months = calculate_total_experience(pdf_text)
            st.write(f"**Total Work Experience:** {years} years and {months} months")

            vector_store = create_vector_store(pdf_text)
            if vector_store:
                query = st.text_input("Ask a question about the document:")
                if query:
                    try:
                        with st.spinner("Finding the answer..."):
                            # Retrieve the most relevant context
                            retrieved_context = vector_store.similarity_search(query, k=1)
                            if not retrieved_context:
                                st.warning("No relevant context found in the document. Please try a different question.")
                            else:
                                context = retrieved_context[0].page_content
                                st.write("**Retrieved Context:**", context)

                                # Split context if too large
                                if len(context) > 512:
                                    st.warning("Context is too large; truncating to fit the model's input size.")
                                    context = context[:512]

                                # Use Hugging Face QA model directly
                                qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
                                response = qa_model({"question": query, "context": context})

                                if response and "answer" in response:
                                    st.write("**Answer:**", response["answer"])
                                else:
                                    st.write("I'm sorry, I couldn't find an answer to your question.")
                    except Exception as e:
                        st.error(f"Error during QA processing: {e}")

# Error handling for no file uploaded
if not uploaded_file:
    st.info("Please upload a PDF file to get started.")
