from langchain_ollama import ChatOllama

chat = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    seed=365,
)

response = chat.invoke(
    """recientemente adopte un perro, cual nombre me sugieres?"""
)

print(response)