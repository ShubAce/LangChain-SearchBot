import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun

from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("Langchain chat with search")

st.sidebar.title("Settings")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hi, I am a chatbot who can search the web. how can i help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"]) 

if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(api_key=groq_api , model="llama3-8b-8192",streaming=True)
    tools = [wiki,search]

    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
