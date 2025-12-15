import streamlit as st
import sys
import os

# Add src to path to import model modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.chatbot import ask_chatbot

st.set_page_config(page_title="BrAinLight Autism Support", page_icon="🧩")

st.title("🧩 BrAinLight Autism Support Chatbot")
st.markdown("""
Welcome! I am here to support parents of children with autism. 
Ask me anything about autism, parenting tips, or personalized advice.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask_chatbot(prompt)
                # The response might contain the prompt itself if the model is not chat-tuned perfectly or if the decoding includes input.
                # But Qwen-Instruct usually handles this well if the prompt format is correct.
                # However, the current build_prompt includes "Answer:" at the end.
                # We should check if the response duplicates the prompt.
                # The `ask_chatbot` returns `tokenizer.decode(outputs[0], skip_special_tokens=True)`.
                # This usually includes the input prompt for decoder-only models unless `return_full_text=False` is set in pipeline, but here we use `model.generate`.
                # `model.generate` returns the full sequence (input + output).
                # I should strip the input from the response in `chatbot.py` or here.
                # I'll strip it here for now by splitting on "Answer:".
                
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")

