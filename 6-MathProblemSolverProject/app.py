import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain_community.tools import  WikipediaQueryRun 
import os 
from langchain.callbacks import StreamlitCallbackHandler 
from dotenv import load_dotenv
load_dotenv()



st.title("Text to math problem Solver using Google Gemma2")

groqAPI=st.sidebar.text_input(label="Enter Groq API key", type="password")

if not groqAPI:
    st.info("Please add your groq api key to continue")
    st.stop()

if groqAPI:
    ## LLM model
    llm = ChatGroq(model="gemma2-9b-it", groq_api_key = groqAPI)

    ## initilaize the tools

    # Wikipedia wrapper
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250) # will use wikipedia api to conduct searches and fetch page summaries
    #toolWiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    # another method to run wikipedia 
    toolWiki = Tool(
        name = "Wikipedia",
        func = api_wrapper_wiki.run,
        description="A tool for searching the internet for the various information related to topic mentioned in the problem."
    )

    ## Math tool
    math_chain = LLMMathChain.from_llm(llm = llm)
    calculator = Tool(
        name = "Calculator",
        func=math_chain.run,
        description="""Useful for when you need to answer questions 
                    about math. This tool is only for math questions and nothing else. Only input
                    math expressions"""
        )
    
    
    template = """ 
                Your a agent tasked for solving users mathematical question. Logically arrive at a solution and provide 
                details explanation and display it pointwise for the question below. 
                Questions: {question}. 
             """
    prompt = PromptTemplate(template=template, 
                            input_variables=["question"])
    

    # combine all tools into chain
    chain=LLMChain(llm = llm, prompt = prompt)

    # resoning tool -> make tool from chain of LLM and prompt
    reasoning_tool= Tool(
        name ="Reasoning",
        func= chain.run, 
        description="A tool for answering logic-based and reasoning questions."
    )

    # initialize the agents
    assistant_agent = initialize_agent(
        tools=  [reasoning_tool, toolWiki, calculator],
        llm = llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = False,
        handle_parsing_errors =True
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role":"assistant", "content":"Hi, I am a math chat bot who can answer all your math question"}
        ]
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    # get question from user
    question = st.text_area("Enter the question:")

 

    if st.button("Find the answer"):
        if question:
            with st.spinner("Generating response"):
                st.session_state.messages.append({"role":"user","content":question})
                with st.chat_message("user"):
                    st.markdown(question)

                stcb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False) 

                response = assistant_agent.run(st.session_state.messages,  callbacks=[stcb])
                st.session_state.messages.append({"role":"assistant","content":response})
                st.success(response)

        else:
            st.warning("Enter your question")
            


    
    










