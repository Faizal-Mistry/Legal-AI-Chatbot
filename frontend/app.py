import streamlit as st
from components.chatUI import render_chat

from components.upload import render_uploader


st.set_page_config(page_title="AI LEGAL ASSISTANT",layout="wide")
 
st.title("⚖️ Smart Legal Chatbot")


render_uploader()
render_chat()

