import streamlit as st
from rag_query import ask_question

st.set_page_config(page_title="Walmart AI Chatbot", layout="wide")

# ------------------------
# Dark theme CSS
# ------------------------
st.markdown(
    """
    <style>
    /* Overall dark background */
    .stApp {
        background-color: #0f111a;
        color: #e5e5e5;
    }
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 10px;
        font-family: 'Segoe UI', sans-serif;
    }
    /* User message bubble (right) */
    .user-msg {
        background-color: #4f46e5;
        color: white;
        padding: 10px 15px;
        border-radius: 20px;
        margin: 5px 0;
        display: inline-block;
        max-width: 70%;
        float: right;
        clear: both;
    }
    /* Assistant message bubble (left) */
    .assistant-msg {
        background-color: #1f2937;
        color: #e5e5e5;
        padding: 10px 15px;
        border-radius: 20px;
        margin: 5px 0;
        display: inline-block;
        max-width: 70%;
        float: left;
        clear: both;
    }
    /* Clear floats after each message */
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    /* Scrollable chat container */
    .chat-scroll {
        max-height: 600px;
        overflow-y: auto;
        padding-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛒 Walmart AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container()

# Display chat history inside scrollable div
with chat_container:
    st.markdown('<div class="chat-container chat-scroll">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="clearfix"><div class="user-msg">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="clearfix"><div class="assistant-msg">{msg["content"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# User input
user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message immediately
    with chat_container:
        st.markdown(f'<div class="clearfix"><div class="user-msg">{user_input}</div></div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        response = ask_question(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display assistant message
        with chat_container:
            st.markdown(f'<div class="clearfix"><div class="assistant-msg">{response}</div></div>', unsafe_allow_html=True)
