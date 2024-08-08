import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from htmlTemplates import css, bot_template, user_template

import faiss
import numpy as np

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def get_vector_stores(text_chunks):
    model_name = "hkunlp/instructor-xl"
    model = SentenceTransformer(model_name)
    print("Model loaded successfully")
    
    # Encode the text chunks
    embeddings = model.encode(text_chunks)
    print("Embeddings created successfully")

    # Create the FAISS index
    faiss_index = create_faiss_index(embeddings)
    print("FAISS index created successfully")
    
    # Create a simple docstore mapping
    docstore = {i: text_chunks[i] for i in range(len(text_chunks))}
    
    # Create an index_to_docstore_id mapping
    index_to_docstore_id = {i: i for i in range(len(text_chunks))}
    
    # Define the embedding function
    def embedding_function(text):
        return model.encode([text])[0]
    
    # Create a FAISS VectorStore
    faiss_vector_store = FAISS(index=faiss_index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embedding_function)
    return faiss_vector_store

def get_conversation_chain(faiss_vector_store):
    # Use LangChain's new method to load a model
    embeddings = HuggingFaceEmbeddings(model_name="google/flan-t5-xxl")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=embeddings,
        retriever=faiss_vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with PDFs")
    user_question = st.text_input("Ask a question about your PDFs")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your files here", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Munching your request..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text) {uncomment to see the working on app}

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks) {uncomment to see the working on app}

                # Create vector embeddings and FAISS index
                vectorstore = get_vector_stores(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
