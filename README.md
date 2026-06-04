# Deepseek Chat Interface

A simple Streamlit-based chat app that connects to a local Ollama server and uses the `deepseek-llm:7b-chat` model.

## Requirements

- Python 3.9+
- `streamlit`
- `requests`
- Local Ollama API server running on `http://localhost:11434`

## Install

```bash
pip install streamlit requests
```

## Run

```bash
streamlit run chat_deepseek1.py
```

## How it works

- The script sends user messages to Ollama using HTTP POST to `/api/generate`
- Messages are streamed back and displayed in the Streamlit UI
- Conversation history is stored in `st.session_state`

## Notes

- Change the model name in `chat_deepseek1.py` if you want to use a different Ollama model.
- Ensure the Ollama daemon is running locally before starting the Streamlit app.
