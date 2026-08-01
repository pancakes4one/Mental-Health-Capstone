import streamlit as st
from llm_interface import handle_message, WELCOME_MESSAGE

st.set_page_config(page_title="Mental Health Treatment Predictor", page_icon="🧠")
st.title("🧠 Mental Health Treatment Predictor")

# Set up session state (persists across Streamlit reruns, one per browser session)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
if "collected" not in st.session_state:
    st.session_state.collected = {}
if "pending_fields" not in st.session_state:
    st.session_state.pending_fields = None

# Show the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle new input
user_input = st.chat_input("Type your answer here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text, st.session_state.collected, prediction, st.session_state.pending_fields = handle_message(
                user_input, st.session_state.collected, st.session_state.pending_fields
            )
        st.write(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Sidebar: reset button + debug view of collected answers
with st.sidebar:
    st.subheader("Session")
    if st.button("Start over"):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
        st.session_state.collected = {}
        st.session_state.pending_fields = None
        st.rerun()

    if st.session_state.collected:
        st.subheader("Collected so far")
        st.json(st.session_state.collected)