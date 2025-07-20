# chatbot.py

import streamlit as st
from helperDemo import init_cohere, ask_cohere

def chatbot_ui(df):
    st.markdown("---")
    st.markdown("### 💬 Chatbot Assistant")

    with st.expander("🤖 Ask Questions About the Chat"):
        if df is not None and not df.empty:
            prompt = st.text_input("Your Question", placeholder="e.g., When were assignments discussed?")
            if prompt:
                with st.spinner("Thinking..."):
                    try:
                        # Shorten context for token limit
                        context = "\n".join((df['Sender'] + ": " + df['Message']).tolist())[:3000]
                        model = init_cohere()
                        response = ask_cohere(model, prompt, context)
                        st.write(response)
                    except Exception as e:
                        st.error(f"❌ Cohere Error: {e}")
        else:
            st.warning("⚠️ Please upload and preprocess a chat file.")
