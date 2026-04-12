import streamlit as st
from pathlib import Path
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import requests
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv() 

os.environ["Groq_Api_key"] = os.getenv("Groq_Api_key")

llm = init_chat_model("groq:qwen/qwen3-32b")

st.set_page_config(page_title="Langchain: Chat with SQL DB", page_icon="🦜🔗")
st.title("🦜🔗 Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLite 3 Database- student.db","Connect to your MySql database"]

selected_opt = st.sidebar.radio(label="choose the db which you want", options = radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MYSQL user")
    mysql_password = st.sidebar.text_input("MYSQL password", type = "password")
    mysql_db = st.sidebar.text_input("MySql database")
else:
    db_uri = LOCALDB
    
    
if not db_uri:
    st.info("Please Enter the database information and Uri")
    

# @st.cache(ttl = "2h")
def configure_db(db_uri,mysql_host = None, mysql_user = None, mysql_password = None, mysql_db = None):
    if db_uri == LOCALDB:
        db_file_path = (Path(__file__).parent/"student.db").absolute()
        print(db_file_path)
        creator = lambda: sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
        return SQLDatabase.from_uri("sqlite:///")
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please Provide all MYSQL connection details")
            st.stop()
        return SQLDatabase.from_uri("mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:port/{mysql_db}")
    
if db_uri == MYSQL:
    db = configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db = configure_db(db_uri)
    
#toolkit
toolkit=SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

agent = create_agent(
    llm,
    tools = tools
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role":"assistant", "content":"How can i help you?"}]
    
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.invoke(user_query, callbacks = [streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
    
    




















