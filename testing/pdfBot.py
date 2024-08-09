import os
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
import tempfile

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from htmlTemplates import css, user_template, bot_template
from langchain_huggingface import HuggingFaceEmbeddings

# Apply necessary patches for async operations in Streamlit
nest_asyncio.apply()

# Load environment variables (for API tokens, etc.)
load_dotenv()

# Function to retrieve and process webpage content
def get_webpage_content(url):
    loader = AsyncHtmlLoader(url)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    return docs_transformed[0].page_content

# Function to retrieve and process PDF content
def get_pdf_content(uploaded_file):
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load the PDF content using PyPDFLoader
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    
    # Clean up the temporary file if needed
    os.remove(tmp_file_path)
    
    return docs[0].page_content

# Function to split the text into smaller, manageable chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    return splitter.split_text(text)

# Function to create a vector store for the text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational retrieval chain
def get_conversation_chain(vectorstore):
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
        },
        huggingfacehub_api_token=hf_token
    )
    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(), 
        verbose=False
    )
    return conversation_chain

# Function to handle user input and display the chat history
def handle_userinput(user_question):
    if st.session_state.conversation:
        result = st.session_state.conversation({'query': user_question})
        
        # Extracting the helpful answer from the result
        result_text = result.get('result', '')
        # Find the start of "Helpful Answer:"
        start_idx = result_text.find('Helpful Answer:')
        if start_idx != -1:
            # Extract the helpful answer
            answer = result_text[start_idx + len('Helpful Answer:'):].strip()
            # Find where the answer ends if needed (in case there's additional text)
            end_idx = answer.find('Question:')  # or other delimiters if present
            if end_idx != -1:
                answer = answer[:end_idx].strip()
        else:
            answer = 'Answer not found in the response.'

        # Append the user and bot messages to the chat history
        st.session_state.chat_history.insert(0, ('bot', answer))
        st.session_state.chat_history.insert(0, ('user', user_question))

    else:
        st.error("Conversation object is not initialized.")

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == 'user':
            st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
        elif role == 'bot':
            st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat with Webpage or PDF", page_icon=":globe_with_meridians:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with a Webpage or PDF :globe_with_meridians:")

    user_question = st.text_input("Ask a question about the document:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Document")
        
        # Webpage input
        url = st.text_input("Enter the URL of the webpage:")
        if st.button("Submit URL"):
            with st.spinner("Munching your request..."):
                # Fetch and process the webpage content
                raw_text = get_webpage_content(url)

                # Split the content into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create a vector store from the text chunks
                vectorstore = get_vectorstore(text_chunks)

                # Initialize the conversational retrieval chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
        # PDF input
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file is not None:
            with st.spinner("Processing the PDF..."):
                # Fetch and process the PDF content
                raw_text = get_pdf_content(uploaded_file)

                # Split the content into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create a vector store from the text chunks
                vectorstore = get_vectorstore(text_chunks)

                # Initialize the conversational retrieval chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
        
        # Add the reset button
        if st.button("Reset"):
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.url = None
            st.session_state.question = ""  # Clear the question field
            st.rerun()  # Use Streamlit's rerun method

if __name__ == '__main__':
    main()
