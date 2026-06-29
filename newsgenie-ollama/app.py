import streamlit as st
from graph import app
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Configuration ---
st.set_page_config(
    page_title="NewsGenie Local AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 NewsGenie Local: AI-Powered News & Info Assistant")
st.markdown("""
This is the **Local Version** of NewsGenie, powered by **Ollama**.
- **News**: Fetches real-time data from NewsAPI.
- **General**: Uses a local LLM for responses.
""")

# --- Sidebar for Category Selection ---
with st.sidebar:
    st.header("Local Settings")
    category = st.selectbox(
        "Select News Category",
        options=["general", "technology", "business", "sports", "entertainment", "health", "science"],
        index=0
    )
    st.info(f"Current category: **{category}**")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Local Genie is thinking..."):
            try:
                inputs = {
                    "messages": [HumanMessage(content=prompt)],
                    "category": category
                }
                final_state = app.invoke(inputs)
                response_content = final_state["messages"][-1].content
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
            except Exception as e:
                error_msg = f"⚠️ Local AI Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
