import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from dotenv import load_dotenv
import os
load_dotenv()

# langsmith tracking
#https://smith.langchain.com/
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")



## prompt template
prompt = ChatPromptTemplate.from_messages([
        ("system","""You are a helpful assistant, please reponse to the user query.
                    give the response in 3 to 4 senteces"""),
        ("user","{que}")
    ]
)
 

def generateResponse(question, model, temp, max_token):
    os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=model,temperature=temp, max_tokens=max_token)
    outParser=StrOutputParser()

    chain = prompt| llm|outParser

    return chain.invoke({"que":question})


st.title("QnA chatbot")
st.sidebar.title("settings")
temp=st.sidebar.slider("temparature",min_value=0.0, max_value=1.0, value=0.7)
maxToken=st.sidebar.slider("maxTokens",min_value=50, max_value=1000, value=500)

# sidebar to select various openSource models
model_name = st.sidebar.selectbox("Select an OpenSource model",
                                  ["llama3-8b-8192","llama3-70b-8192","llama-guard-3-8b",
                                          "gemma-7b-it","gemma2-9b-it"])



## user input
st.write("Ask question to llm")
userInput=st.text_input("You:")

if userInput:
    response = generateResponse(userInput, model_name, temp, maxToken)
    st.write(response)

else:
    st.write("Please provide the user query")