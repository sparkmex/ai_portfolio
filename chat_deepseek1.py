"""Streamlit chat interface for Ollama Llama3 models.

This script connects to a local Ollama API server and sends user prompts to
an LLM model. It renders a simple chat UI using Streamlit and keeps the
conversation history in session state.

Usage:
    streamlit run chat_deepseek1.py

Requirements:
    pip install streamlit requests
    Ollama server running locally on port 11434
"""

import json

import requests
import streamlit as st


def ollama_chat(prompt: str, model: str = "deepseek-llm:7b-chat") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
    )
    result = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            result += data.get("response", "")
    return result


st.title("Chat con Ollama (Llama3)")

if "history" not in st.session_state:
    st.session_state.history = []

if "input_box" not in st.session_state:
    st.session_state.input_box = ""

def send_message() -> None:
    """Process the user message and append both sides of the dialogue."""
    user_input = st.session_state.input_box
    if not user_input:
        return

    st.session_state.history.append({"role": "user", "content": user_input})
    with st.spinner("Llama is thinking..."):
        llama_response = ollama_chat(user_input)

    st.session_state.history.append({"role": "llama", "content": llama_response})
    st.session_state.input_box = ""  # Clear the input field.

col1, col2, col3 = st.columns([2, 1, 5])

with col1:
    st.text_input(
        "Type your message:",
        key="input_box",
        on_change=send_message
    )

with col3:
    # Invertimos el historial para mostrar las nuevas conversaciones arriba
    for msg in reversed(st.session_state.history):
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style='text-align: left; background-color: #e6f7ff; padding: 8px; border-radius: 8px; margin-bottom: 5px; width: 70%;'>
                    <b>You:</b> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='text-align: right; background-color: #fffbe6; padding: 8px; border-radius: 8px; margin-bottom: 5px; width: 70%; float: right;'>
                    <b>Llama:</b> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )