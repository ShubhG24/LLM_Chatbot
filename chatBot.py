import os
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
import tempfile
import json

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from htmlTemplates import css, user_template, bot_template
from langchain_huggingface import HuggingFaceEmbeddings

nest_asyncio.apply()
load_dotenv()

# URL content
def get_webpage_content(url):
    loader = AsyncHtmlLoader(url)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    return docs_transformed[0].page_content

# PDF content
def get_pdf_content(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    os.remove(tmp_file_path)
    return docs[0].page_content

# JSON content
def get_json_content(uploaded_file):
    json_data = json.load(uploaded_file)
    json_text = json.dumps(json_data, indent=4)
    return json_text

# Chunking
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    return splitter.split_text(text)

# Vector Store
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Chain
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

# Handling the user inputs
def handle_userinput(user_question):
    if st.session_state.conversation:
        result = st.session_state.conversation({'query': user_question})
        
        result_text = result.get('result', '')
        # Extracting from res o/p "Helpful Answer:"
        start_idx = result_text.find('Helpful Answer:')
        if start_idx != -1:
            answer = result_text[start_idx + len('Helpful Answer:'):].strip()
            end_idx = answer.find('Question:')  
            if end_idx != -1:
                answer = answer[:end_idx].strip()
        else:
            answer = 'Answer not found in the response.'

        # Append messages to the chat history
        st.session_state.chat_history.insert(0, ('bot', answer))
        st.session_state.chat_history.insert(0, ('user', user_question))

    else:
        st.error("Conversation object is not initialized.")

    # Display user/bot responses
    for role, message in st.session_state.chat_history:
        if role == 'user':
            st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
        elif role == 'bot':
            st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)

# GUI
def main():
    st.set_page_config(page_title="Chat with Multiple Documents", page_icon=":globe_with_meridians:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with Multiple Documents :globe_with_meridians:")

    user_question = st.text_input("Ask a question about the documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        
        # URL inputs
        urls = st.text_area("Enter URLs of webpages (one per line):")
        url_list = urls.splitlines()

        # PDF inputs
        uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

        # JSON inputs
        uploaded_jsons = st.file_uploader("Upload JSON files", type="json", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Munching your request..."):
                combined_text = ""

                # Process URLs
                for url in url_list:
                    combined_text += get_webpage_content(url) + "\n"

                # Process PDFs
                for uploaded_pdf in uploaded_pdfs:
                    combined_text += get_pdf_content(uploaded_pdf) + "\n"

                # Process JSONs
                for uploaded_json in uploaded_jsons:
                    combined_text += get_json_content(uploaded_json) + "\n"

                # embeddings
                text_chunks = get_text_chunks(combined_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
        
        

if __name__ == '__main__':
    main()
