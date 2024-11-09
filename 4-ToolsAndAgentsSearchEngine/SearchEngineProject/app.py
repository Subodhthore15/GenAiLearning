import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun 
from langchain.callbacks import StreamlitCallbackHandler # tools can communicate within themselves
from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")

# Wikipedia wrapper
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250) # will use wikipedia api to conduct searches and fetch page summaries
toolWiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
# Arxiv wrapper
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250) 
toolArxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

toolGoogleFinance = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper(), name="Google search")


search = DuckDuckGoSearchRun(name="DuckDuckGo Search") # search from internet


st.title("Search Engine")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant", 
                                     "content":"I am a chatbot who can search on web"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input(placeholder="Enter your query")

if user_prompt:
    st.session_state.messages.append({"role":"user","content":user_prompt})
    # display user message in chat history container

    with st.chat_message("user"):
        st.markdown(user_prompt)

    ## LLM model
    llm = ChatGroq(model="Llama3-8b-8192",streaming=True)

    # initialize all the tools 
    tools = [toolGoogleFinance,toolWiki, toolArxiv, search]

    # agents
    agents = initialize_agent(tools=tools, llm = llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                              handle_parsing_errors=True )

    with st.chat_message("assistant"):
        # it shows how it is generating response
        stcb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False) 

        response = agents.run(st.session_state.messages,  callbacks=[stcb])
        st.session_state.messages.append({"role":"assistant","content":response})

        st.write(response)







