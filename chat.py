from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

historial = [
    {"role": "system", "content": "Eres un asistente útil y conciso."}
]

print("Chat con deepseek-llm:7b-chat (escribe 'salir' para terminar)\n")

while True:
    user_input = input("Tú: ")
    if user_input.lower() == "salir":
        break

    historial.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="deepseek-llm:7b-chat",
        messages=historial,
        temperature=0.7
    )

    respuesta = response.choices[0].message.content
    print(f"DeepSeek: {respuesta}\n")

    historial.append({"role": "assistant", "content": respuesta})