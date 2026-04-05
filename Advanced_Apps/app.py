import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import embeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_huggingface import HuggingFaceEmbeddings


# from emmedinggs import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")



load_dotenv()

os.environ['HF_Token']=os.getenv("HF_Token")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    max_retries=2
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Strictly Provide the most accurate response based on the question 
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        print("1")
        st.session_state.loader = PyPDFDirectoryLoader("F:\GenAI\Advanced_Apps")
        print("2")
        st.session_state.docs = st.session_state.loader.load()
        print("3")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        print("4")
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:10])
        print("5")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        print("6")
        


user_prompt= st.text_input("Enter your query")

if st.button("Docuemnt Embedding"):
    create_vector_embeddings()
    st.write("Vector Database is Ready!!!!")
    
import time


if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain= create_retrieval_chain(retriever, document_chain)
    
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    
    
    st.write(response['answer'])
    
    with st.expander("Document similarity search"):
        for i,doc in enumerate (response["context"]):
            st.write(doc.page_content)
            st.write("----------------------")