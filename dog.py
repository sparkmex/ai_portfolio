from langchain_ollama import ChatOllama

chat = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    seed=365,
)

response = chat.invoke(
    """I have recently adopted a dog.
Could you suggest some dog names?"""
)

print(response)