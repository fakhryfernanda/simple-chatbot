from dotenv import load_dotenv
import os

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY")

# Upload PDF files
st.header("My First chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\n",
        chunk_size = 1000,     # character counts
        chunk_overlap = 150,   # bring 150 characters from previous chunk
        length_function = len
    )

    chunks = text_splitter.split_text(text)

    # generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user questions
    user_question = st.text_input("Type your question here")
    
    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)

        # define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000, # roughly 750 words
            model_name = "gpt-3.5-turbo"
        )

        # output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)
















