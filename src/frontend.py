import streamlit as st

from rag_client import RagClient

client = RagClient()

with st.sidebar:
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by a Local LLM with lmstudio")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.query_rag(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
