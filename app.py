import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import tempfile
from PyPDF2 import PdfReader

load_dotenv()

# Initialize environment variables
groq_api_key = "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"
os.environ["GOOGLE_API_KEY"] ="AIzaSyCHiFTOdWEmKKJACV3wesTqWNiy591TgmE"

st.title("Book Summarizer with Chapters")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Prompt for summarization
prompt = ChatPromptTemplate.from_template(
    """
You are a skilled book summarizer. Your task is to:
1. Extract key concepts and rules mentioned in the book.
2. Explain each concept with clear examples.

Write in plain language, suitable for a general audience, and focus on presenting the chapter's essence in a concise and clear manner.
<Context>
{context}
</Context>
"""
)

def process_pdf(uploaded_file, chapter_info):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    pdf_reader = PdfReader(tmp_file_path)
    temp_folder = tempfile.mkdtemp()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    summaries = {}
    chapter_texts = {}
    chapter_embeddings = {}

    # Process each chapter based on user input
    for chapter, page_range in chapter_info.items():
        st.write(f"Processing {chapter}...")

        # Extract text for the chapter
        chapter_text = ""
        for page in range(page_range[0] - 1, page_range[1]):  # Adjusting for 0-based index
            chapter_text += pdf_reader.pages[page].extract_text()

        chapter_texts[chapter] = chapter_text

        # Save chapter as a text file
        chapter_file = os.path.join(temp_folder, f"{chapter}.txt")
        with open(chapter_file, "w", encoding="utf-8") as f:
            f.write(chapter_text)

        # Embed the chapter
        chapter_embedding = FAISS.from_texts([chapter_text], embeddings)
        chapter_embeddings[chapter] = chapter_embedding

        # Summarize the chapter
        retriever = chapter_embedding.as_retriever()
        retrieved_docs = retriever.get_relevant_documents("Extract key concepts and rules with examples")
        response = document_chain.invoke({"context": " ".join([doc.page_content for doc in retrieved_docs])})
        summaries[chapter] = response

    # Combine summaries into a single file
    combined_summary = "\n\n".join([f"{chapter}:\n{summary}" for chapter, summary in summaries.items()])
    combined_file = os.path.join(temp_folder, "combined_summary.txt")
    with open(combined_file, "w", encoding="utf-8") as f:
        f.write(combined_summary)

    return combined_summary, temp_folder

# Streamlit interface
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
num_chapters = st.number_input("Enter the number of chapters in the book:", min_value=1, step=1)

chapter_info = {}
if num_chapters:
    st.write("Specify the page range for each chapter:")
    for i in range(1, num_chapters + 1):
        start_page = st.number_input(f"Start page for Chapter {i}:", min_value=1, step=1)
        end_page = st.number_input(f"End page for Chapter {i}:", min_value=start_page, step=1)
        chapter_info[f"Chapter {i}"] = (start_page, end_page)

if st.button("Process and Summarize"):
    if uploaded_file and chapter_info:
        combined_summary, temp_folder = process_pdf(uploaded_file, chapter_info)
        st.success("Summarization Complete!")
        st.text_area("Combined Summary", combined_summary, height=400)

        # Provide download link for the combined summary
        combined_file = os.path.join(temp_folder, "combined_summary.txt")
        with open(combined_file, "rb") as f:
            st.download_button("Download Combined Summary", f, file_name="combined_summary.txt")
    else:
        st.error("Please upload a PDF file and specify chapter page ranges.")
