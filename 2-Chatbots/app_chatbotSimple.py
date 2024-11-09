import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
load_dotenv() 


os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

if 'store'  not in st.session_state:
    st.session_state.store = {}

def get_session_history(sessionId:str) -> BaseChatMessageHistory:
    if sessionId not in st.session_state.store:
        st.session_state.store[sessionId] = ChatMessageHistory()
    return st.session_state.store[sessionId]

from langchain_groq import ChatGroq
llm=ChatGroq(model="Gemma2-9b-It")


sessionUser=st.text_input("Enter the session id")

if sessionUser:
    user_question = st.text_input("Enter the question you want to ask?")
    if user_question:
        with_message_history = RunnableWithMessageHistory(llm, get_session_history)
        config = {"configurable":{"session_id":sessionUser}}

        response=with_message_history.invoke(input={"input":user_question}, config=config)
        st.write(response.content)

