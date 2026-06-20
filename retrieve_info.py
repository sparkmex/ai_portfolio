"""
Consulta interactiva sobre el contenido de una página web usando LangChain + RAG.

Stack 100% local:
    - Carga de contenido:  WebBaseLoader (LangChain)
    - Embeddings:          nomic-embed-text (vía Ollama)
    - Vector store:        Chroma (en memoria)
    - LLM de respuesta:    qwen3.5:9b (vía Ollama)

Requisitos previos:
    ollama pull nomic-embed-text
    ollama pull deepseek-llm:7b-chat
    ollama pull qwen3.5:9b  # funciona mejor este modelo que deepseek-llm:7b-chat para preguntas basadas en contexto, aunque es más lento
    ollama pul mistral:7b-instruct

Instalación de dependencias:
    pip install langchain langchain-community langchain-ollama chromadb beautifulsoup4

Uso:
    python3 chat_web.py https://ejemplo.com/articulo
"""

import sys

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


MODELO_EMBEDDINGS = "nomic-embed-text"
MODELO_LLM = "mistral:7b-instruct"

# Prompt estricto: obliga al modelo a responder SOLO con base en el contexto
# recuperado del sitio web, y a admitir cuando no tiene la información,
# en vez de inventar (alucinar).
PLANTILLA_PROMPT = """Eres un asistente que responde preguntas ÚNICAMENTE basándote en el
siguiente contexto extraído de una página web. No uses conocimiento externo ni
información que no esté en el contexto.

Si la respuesta no se encuentra en el contexto, responde exactamente:
"No encuentro esa información en el contenido de esta página."

No inventes datos, nombres, cifras ni hechos que no aparezcan en el contexto.

Contexto:
{context}

Historial de la conversación:
{chat_history}

Pregunta: {question}

Respuesta (basada solo en el contexto anterior):"""

QA_PROMPT = PromptTemplate(
    template=PLANTILLA_PROMPT,
    input_variables=["context", "chat_history", "question"]
)


def cargar_pagina(url: str):
    """Descarga y parsea el contenido de la URL dada."""
    print(f"Cargando contenido de: {url}")
    loader = WebBaseLoader(url)
    documentos = loader.load()
    print(f"Se cargaron {len(documentos)} documento(s) "
          f"({sum(len(d.page_content) for d in documentos)} caracteres en total).")
    return documentos


def dividir_en_chunks(documentos):
    """Divide el contenido en fragmentos pequeños para indexar mejor."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documentos)
    print(f"Contenido dividido en {len(chunks)} fragmentos.")
    return chunks


def crear_vectorstore(chunks):
    """Genera embeddings de los fragmentos y los guarda en un vector store en memoria."""
    print("Generando embeddings (esto puede tardar un poco)...")
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("Índice listo.\n")
    return vectorstore


def crear_cadena_qa(vectorstore):
    """Crea la cadena conversacional con memoria, recuperación (RAG) y prompt estricto."""
    llm = ChatOllama(model=MODELO_LLM, temperature=0.0)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    memoria = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memoria,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    return qa_chain


UMBRAL_RELEVANCIA = 0.99  # distancia máxima aceptable (menor = más parecido, depende del embedding)
MENSAJE_NO_ENCONTRADO = "No encuentro esa información en el contenido de esta página."
MODO_DEBUG = True  # muestra el score real de similitud para calibrar el umbral


def es_pregunta_relevante(vectorstore, pregunta: str) -> bool:
    """Verifica si la pregunta tiene fragmentos suficientemente relacionados en el sitio.

    Usa similarity_search_with_score: en Chroma, un score MÁS BAJO significa
    MÁS similitud (es una distancia, no un porcentaje de parecido).
    """
    resultados = vectorstore.similarity_search_with_score(pregunta, k=4)
    if not resultados:
        return False

    mejor_score = min(score for _, score in resultados)

    if MODO_DEBUG:
        print(f"   [debug] mejor score de similitud: {mejor_score:.4f} "
              f"(umbral actual: {UMBRAL_RELEVANCIA})")

    return mejor_score <= UMBRAL_RELEVANCIA


def chat_interactivo(qa_chain, vectorstore):
    """Loop de preguntas y respuestas sobre el contenido indexado."""
    print("=" * 60)
    print("Chat listo. Pregunta lo que quieras sobre la página.")
    print("Escribe 'salir' para terminar.")
    print("=" * 60 + "\n")

    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break
        if not pregunta:
            continue

        if not es_pregunta_relevante(vectorstore, pregunta):
            print(f"\n{MODELO_LLM}: {MENSAJE_NO_ENCONTRADO}\n")
            continue

        resultado = qa_chain.invoke({"question": pregunta})
        respuesta = resultado["answer"]

        print(f"\n{MODELO_LLM}: {respuesta}\n")


def main():
    if len(sys.argv) < 2:
        print(f"Uso: python3 {sys.argv[0]} <url>")
        sys.exit(1)

    url = sys.argv[1]

    documentos = cargar_pagina(url)
    chunks = dividir_en_chunks(documentos)
    vectorstore = crear_vectorstore(chunks)
    qa_chain = crear_cadena_qa(vectorstore)

    chat_interactivo(qa_chain, vectorstore)


if __name__ == "__main__":
    main()