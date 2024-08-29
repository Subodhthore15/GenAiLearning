import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,WebBaseLoader
import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
## LLM model
llm = ChatGroq(model="Llama3-8b-8192")

## streamlit app

st.set_page_config(page_title="Langchain: Summarize text from YouTube or Website")
st.title("Langchain: Summarize text from YouTube or Website")

## Prompt template
template = """ 
 Provide Summary of the following content in 150 words. 
Content: \n {text}. 
"""
prompt = PromptTemplate(template = template, input_variables = ["text"])


## Get the URL to be summarized
URL_input = st.text_input("Enter the URL")

if st.button("Summarize the content for URL"):

    if not URL_input:
        st.error("Please provide the information URL to get started.")

    elif not validators.url(URL_input):
        st.error("Please provide valid URL.")

    else: # valid url 
        with st.spinner("waiting..."):
            ## loading the URL data 
            if "youtube.com" in URL_input: # youtube url
                loader = YoutubeLoader.from_youtube_url(URL_input, add_video_info = True)
                

            else: # other website url
                loader = WebBaseLoader(  web_paths=[URL_input] )

            docs = loader.load()

            # chain for Summarization; 
            chain = load_summarize_chain(llm = llm, chain_type = "stuff", prompt = prompt)
            output=chain.run(docs)
        
            st.success(output)
                








