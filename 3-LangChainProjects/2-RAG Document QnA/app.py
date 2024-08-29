import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

## what does this do?
from langchain.chains.combine_documents import create_stuff_documents_chain  
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()
## load the groq api 
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
## LLM model
llm = ChatGroq(model="llama3-8b-8192")

## prompts
prompt = ChatPromptTemplate.from_template(
    """
        Answer the questions based on the provided context only. 
        Please provide the most accurate response based on the context. 
        <context>
        {context}
        </context>


        Question: {input}
    """
)


def create_vectorEmbeddings():
    # use session for storing embeddings
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="gemma2:2b")
        st.session_state.loader = PyPDFDirectoryLoader("pdfsData") # data ingestion
        st.session_state.docs = st.session_state.loader.load() # documents loading
        st.session_state.textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.finalDocumentsChunks = st.session_state.textSplitter.split_documents(st.session_state.docs)
        st.session_state.vectorsDB = FAISS.from_documents(st.session_state.finalDocumentsChunks, 
                                                          st.session_state.embeddings)
    
st.title("Personal finance Query. RAG pdf")
userInput = st.text_input("Enter your query about finance data")

if st.button("Document embeddings"):
    create_vectorEmbeddings()
    st.write("Vector DB is ready with embeddings")



if userInput:
    # 2] give contextual info to llm
    documentChain=create_stuff_documents_chain(llm=llm, prompt=prompt) 
    retriever= st.session_state.vectorsDB.as_retriever(
                                            search_type = "similarity",
                                            search_kwargs={"k":1})

    # 1] from vector db get contextual info and set as a context var
    ragChain = create_retrieval_chain(retriever, documentChain)

    response=ragChain.invoke({"input":userInput})
    st.write(response["answer"])

    # streamlit expandar to search similar document
    with st.expander("Documents similarity for user input"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("###########################################")



