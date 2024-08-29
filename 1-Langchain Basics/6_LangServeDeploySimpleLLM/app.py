from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

LLM=ChatGroq(model='gemma2-9b-it', temperature= 0.1)

# create prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system","Translate the following into {language} language"),
    ("user","{text}")
])

parser=StrOutputParser()

# chain
chain= prompt|LLM|parser

# app defination
app = FastAPI(title="langchain app LLM", version="1.0", description="Simple app LLM using langchain interface")


# adding chain route
add_routes(
    app, 
    chain,
    path='/chain'
)


if __name__ =="__main__":
    import uvicorn 
    uvicorn.run(app, host='localhost')


