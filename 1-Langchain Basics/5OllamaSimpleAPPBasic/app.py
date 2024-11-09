
import streamlit as st
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

 
# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a math assistant for student. Please give correct answer"),
        ("user","Question:{que}")
    ]

)


llm = Ollama(model = 'gemma2:2b')

parser=StrOutputParser()

st.title("Math teacher: with gemma2 model")

user_input = st.text_input("Ask question about math")

if user_input:
    chain = prompt|llm|parser
    st.write(chain.invoke({"que":user_input}))









