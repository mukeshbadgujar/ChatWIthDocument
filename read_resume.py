import streamlit as st
import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Load Hugging Face models for QA and summarization
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=-1)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""  # Handle pages with no text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Function to create a vector store for text retrieval
def create_vector_store(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)

        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Function to build the QA chain
def build_qa_chain(vector_store):
    try:
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

# Streamlit UI
st.title("Resume-Friendly QA Application")

uploaded_file = st.file_uploader("Upload a Resume (PDF)", type="pdf")
if uploaded_file:
    with st.spinner("Processing the resume..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            # Limit text size for large resumes
            if len(pdf_text) > 1_000_000:  # ~1 million characters
                st.warning("The uploaded file is too large. Only the first part of the resume will be processed.")
                pdf_text = pdf_text[:1_000_000]

            vector_store = create_vector_store(pdf_text)
            if vector_store:
                qa_chain = build_qa_chain(vector_store)
                if qa_chain:
                    query = st.text_input("Ask a question about the resume:")
                    if query:
                        with st.spinner("Finding the answer..."):
                            try:
                                # Retrieve relevant context
                                retrieved_context = vector_store.similarity_search(query, k=1)
                                if not retrieved_context:
                                    st.warning("No relevant context found. Please try a different question.")
                                else:
                                    context = retrieved_context[0].page_content
                                    # Truncate context if too large for the model
                                    if len(context) > 512:
                                        context = context[:512]

                                    # Use QA chain to answer the query
                                    response = qa_chain.run(query)
                                    if response.strip():
                                        st.write("**Answer:**", response)
                                    else:
                                        st.write("I'm sorry, I couldn't find an answer to your question.")
                            except Exception as e:
                                st.error(f"Error during QA processing: {e}")

                    # Summarization Section
                    st.markdown("### Resume Summary:")
                    try:
                        summary = summarizer(pdf_text, max_length=300, min_length=100, do_sample=False)
                        st.write(summary[0]["summary_text"])
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")

else:
    st.info("Please upload a resume in PDF format to begin.")
