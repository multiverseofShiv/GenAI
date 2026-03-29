from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_retries=2
)

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


system_template = "Translate the following into {Language}"
prompt_template = ChatPromptTemplate.from_messages([("system",system_template),("user","{text}")])

parser = StrOutputParser()


#create chain

chain = prompt_template | model | parser


#App Definition

app = FastAPI(title="Langchain Server",
              version="1.0",
              description="A Simple API server using Langchain ruunable interface")

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)



