import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_retries=2
)

os.environ['Api_key']=os.getenv("HF_Token")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


st.title("Welcome to the world of ai")

input_text = st.text_input("Enter your query")



OutputParser = StrOutputParser()





prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are an expert on all topics answer the question"),
        ("user","{input}")
        
    ]
)

chain = prompt|model|OutputParser


response = chain.invoke({input_text})

st.write(response)


