"""
Consulta interactiva sobre el contenido de una página web usando LangChain + RAG.

Variante con conversión a Markdown:
    A diferencia de chat_web.py (que indexa el HTML/texto crudo extraído por
    WebBaseLoader), este script primero convierte la página a Markdown limpio
    y SOLO DESPUÉS genera los embeddings y los chunks. Esto reduce ruido
    (menús, footers, scripts) y produce chunks más coherentes al cortar por
    encabezados (#, ##, ###) en vez de por cantidad arbitraria de caracteres.

Pipeline:
    1. Descarga del HTML (requests)
    2. Extracción de contenido principal (trafilatura) -> quita boilerplate
    3. Conversión a Markdown (markdownify) sobre el HTML limpio
    4. Guarda el .md resultante en ./paginas_md/ para inspección física
    5. Split consciente de headers (MarkdownHeaderTextSplitter)
    6. Split adicional por tamaño (RecursiveCharacterTextSplitter) por si
       una sección quedó muy larga
    7. Embeddings + Chroma + cadena conversacional (igual que el original)

Stack 100% local:
    - Extracción de contenido:  trafilatura
    - Conversión a Markdown:    markdownify
    - Embeddings:               nomic-embed-text (vía Ollama)
    - Vector store:             Chroma (en memoria)
    - LLM de respuesta:         qwen3.5:9b (vía Ollama)

Requisitos previos:
    ollama pull nomic-embed-text
    ollama pull qwen3.5:9b

Instalación de dependencias:
    pip install langchain langchain-community langchain-ollama chromadb \
                requests trafilatura markdownify

Uso:
    python3 chat_web_md.py https://ejemplo.com/articulo
"""

import sys
import threading
import time
import itertools
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura
from markdownify import markdownify as html_a_md

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document


MODELO_EMBEDDINGS = "nomic-embed-text"
MODELO_LLM = "mistral:7b-instruct"

# Carpeta donde se guardan los .md generados, para poder revisarlos
# físicamente y verificar qué se está indexando realmente.
CARPETA_MD = Path("paginas_md")

HEADERS_HTTP = {
    "User-Agent": "Mozilla/5.0 (compatible; chat_web_md/1.0)"
}

# Encabezados Markdown por los que se va a partir el contenido antes del
# split por tamaño. Cada tupla es (marcador_md, nombre_de_metadato).
NIVELES_HEADER = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

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


class Spinner:
    """Muestra un indicador de actividad en la terminal mientras una
    operación bloqueante (como qa_chain.invoke) está corriendo en otro hilo.

    Uso:
        with Spinner("Pensando"):
            resultado = qa_chain.invoke(...)
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, mensaje: str = "Pensando", intervalo: float = 0.1):
        self.mensaje = mensaje
        self.intervalo = intervalo
        self._detener = threading.Event()
        self._hilo = None
        self._inicio = None

    def _animar(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._detener.is_set():
                break
            transcurrido = time.time() - self._inicio
            sys.stdout.write(f"\r{frame} {self.mensaje}... ({transcurrido:4.1f}s)")
            sys.stdout.flush()
            time.sleep(self.intervalo)

    def __enter__(self):
        self._inicio = time.time()
        self._detener.clear()
        self._hilo = threading.Thread(target=self._animar, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._detener.set()
        if self._hilo is not None:
            self._hilo.join()
        # Limpia la línea del spinner para que la respuesta no quede pegada
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()


def nombre_archivo_desde_url(url: str) -> str:
    """Genera un nombre de archivo legible a partir de la URL, ej:
    https://ejemplo.com/blog/mi-articulo -> ejemplo.com_blog_mi-articulo.md
    """
    partes = urlparse(url)
    ruta = partes.path.strip("/").replace("/", "_") or "index"
    nombre = f"{partes.netloc}_{ruta}"
    # Quita caracteres problemáticos para nombres de archivo
    nombre = "".join(c if c.isalnum() or c in "-_." else "_" for c in nombre)
    return f"{nombre}.md"


def guardar_markdown(markdown: str, url: str) -> Path:
    """Guarda el Markdown generado en disco, dentro de CARPETA_MD,
    para poder inspeccionarlo físicamente con cualquier editor/visor.
    """
    CARPETA_MD.mkdir(parents=True, exist_ok=True)
    ruta_archivo = CARPETA_MD / nombre_archivo_desde_url(url)
    ruta_archivo.write_text(markdown, encoding="utf-8")
    return ruta_archivo


def descargar_html(url: str) -> str:
    """Descarga el HTML crudo de la URL dada."""
    print(f"Descargando: {url}")
    resp = requests.get(url, headers=HEADERS_HTTP, timeout=20)
    resp.raise_for_status()
    return resp.text


def extraer_contenido_principal(html_crudo: str, url: str) -> str:
    """Usa trafilatura para quedarse solo con el contenido relevante del
    artículo (quita menús, footers, sidebars, scripts, etc.) y devuelve
    el resultado como HTML limpio para luego convertir a Markdown.
    """
    html_limpio = trafilatura.extract(
        html_crudo,
        url=url,
        output_format="html",
        include_comments=False,
        include_tables=True,
    )
    if not html_limpio:
        print("  [aviso] trafilatura no pudo extraer contenido limpio; "
              "se usará el HTML crudo completo como respaldo.")
        return html_crudo
    return html_limpio


def convertir_a_markdown(html_limpio: str) -> str:
    """Convierte el HTML (ya limpio) a Markdown."""
    md = html_a_md(html_limpio, heading_style="ATX")
    # Colapsa líneas en blanco excesivas que deja la conversión
    lineas = [linea.rstrip() for linea in md.splitlines()]
    md_limpio = "\n".join(lineas)
    while "\n\n\n" in md_limpio:
        md_limpio = md_limpio.replace("\n\n\n", "\n\n")
    return md_limpio.strip()


def dividir_en_chunks(markdown: str, fuente_url: str):
    """Divide el Markdown primero por encabezados (#, ##, ###) y luego,
    si alguna sección sigue siendo muy larga, por tamaño de caracteres.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=NIVELES_HEADER,
        strip_headers=False,
    )
    fragmentos_por_header = header_splitter.split_text(markdown)

    if not fragmentos_por_header:
        # Si el markdown no tiene headers, cae aquí como un solo bloque
        fragmentos_por_header = [Document(page_content=markdown, metadata={})]

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
    )
    chunks_finales = size_splitter.split_documents(fragmentos_por_header)

    for chunk in chunks_finales:
        chunk.metadata["source"] = fuente_url

    print(f"Markdown dividido en {len(fragmentos_por_header)} secciones por "
          f"encabezado -> {len(chunks_finales)} chunks finales.")
    return chunks_finales


def crear_vectorstore(chunks):
    """Genera embeddings de los fragmentos y los guarda en un vector store en memoria."""
    print("Generando embeddings (esto puede tardar un poco)...")
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS)
    with Spinner("Generando embeddings"):
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("Índice listo.\n")
    return vectorstore


def crear_cadena_qa(vectorstore):
    """Crea la cadena conversacional con memoria, recuperación (RAG) y prompt estricto."""
    llm = ChatOllama(model=MODELO_LLM, temperature=0.0)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

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


UMBRAL_RELEVANCIA = 1.6  # umbral más flexible para evitar rechazar respuestas válidas
MENSAJE_NO_ENCONTRADO = "No encuentro esa información en el contenido de esta página."
MODO_DEBUG = True  # muestra el score real de similitud para calibrar el umbral


def expandir_consulta(pregunta: str):
    """Genera variantes de la pregunta para mejorar la recuperación de términos clave."""
    texto = pregunta.strip()
    variantes = [texto]
    texto_lower = texto.lower()

    # Variantes útiles para preguntas sobre validez educativa o acreditación.
    if any(palabra in texto_lower for palabra in ("rvoe", "validez", "acredit", "registro")):
        variantes.extend([
            "RVOE",
            "registro de validez oficial",
            "validez oficial",
            "registro oficial de estudios",
            "acreditación estatal",
            "reconocimiento oficial",
        ])

    # Si la pregunta menciona “tiene” o “cuenta con”, también probamos la forma nominal.
    if "tiene" in texto_lower or "cuenta con" in texto_lower:
        variantes.append("cuenta con RVOE")

    # Variantes para programas académicos específicos.
    if "doctorado" in texto_lower:
        variantes.extend([
            "Doctorado en Ciencias de la Educación",
            "doctorado en ciencias de la educación",
            "doctorado en ciencias",
            "ciencias de la educación",
        ])

    return list(dict.fromkeys(variantes))


def _normalizar_texto(texto: str) -> str:
    """Normaliza texto para comparar sin importar mayúsculas, tildes ni puntuación."""
    texto = texto.lower()
    texto = texto.replace("á", "a").replace("é", "e")
    texto = texto.replace("í", "i").replace("ó", "o").replace("ú", "u")
    texto = texto.replace("ñ", "n")
    return re.sub(r"[^a-z0-9\s]", " ", texto)


def _extraer_palabras_clave(texto: str):
    """Extrae términos útiles para reforzar la búsqueda por palabras clave."""
    return [palabra for palabra in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]+", texto.lower()) if len(palabra) >= 3]


def obtener_docs_relevantes(vectorstore, pregunta: str, k: int = 6):
    """Recupera documentos usando variantes textuales + similitud semántica."""
    query_terms = set(_extraer_palabras_clave(pregunta))
    variantes = expandir_consulta(pregunta)
    variantes_norm = [_normalizar_texto(v) for v in variantes]
    resultados = []
    vistos = set()

    # Búsqueda lexical explícita: si la pregunta tiene palabras clave, prioriza chunks que las contengan.
    docs_lex = []
    for doc in vectorstore.similarity_search(pregunta, k=k * 5):
        contenido_norm = _normalizar_texto(doc.page_content)
        hits = sum(1 for term in query_terms if term in _normalizar_texto(doc.page_content))
        if hits > 0 or any(v in contenido_norm for v in variantes_norm):
            docs_lex.append((doc, hits))

    # También agregamos resultados semánticos para completar los casos donde no hay coincidencia exacta.
    for doc, score in vectorstore.similarity_search_with_score(pregunta, k=k * 3):
        contenido = doc.page_content
        contenido_norm = _normalizar_texto(contenido)
        hits = sum(1 for term in query_terms if term in _normalizar_texto(contenido))
        score_lex = 0.0
        if any(v in contenido_norm for v in variantes_norm):
            score_lex = 0.15
        score_hibrido = score - (0.08 * hits) - score_lex
        clave = (doc.metadata.get("source"), contenido[:250])
        if clave not in vistos:
            resultados.append((doc, score_hibrido, score, hits))
            vistos.add(clave)

    # Añade los resultados lexicográficos al inicio del ranking.
    for doc, hits in docs_lex:
        clave = (doc.metadata.get("source"), doc.page_content[:250])
        if clave not in vistos:
            resultados.append((doc, -0.5 - (0.05 * hits), 0.0, hits))
            vistos.add(clave)

    resultados.sort(key=lambda item: item[1])
    return [(doc, score_h, score_orig, hits) for doc, score_h, score_orig, hits in resultados[:k]]


def es_pregunta_relevante(vectorstore, pregunta: str) -> bool:
    """Verifica si la pregunta tiene fragmentos suficientemente relacionados.

    En lugar de usar solo un umbral rígido sobre embeddings, revisa primero
    si el contenido recuperado tiene coincidencias semánticas o textuales.
    """
    resultados = obtener_docs_relevantes(vectorstore, pregunta, k=6)
    if not resultados:
        return False

    mejor_score = min(score_h for _, score_h, _, _ in resultados)

    if MODO_DEBUG:
        print(f"   [debug] mejor score hibrido: {mejor_score:.4f} "
              f"(umbral actual: {UMBRAL_RELEVANCIA})")
        print(f"   [debug] documentos recuperados: {len(resultados)}")

    # Fallback fuerte: si aparece una coincidencia textual exacta, no rechazar.
    if any(hits > 0 for _, _, _, hits in resultados):
        return True

    # Si el score hibrido es razonable aunque no haya exactitud literal, aceptamos.
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

        # Para depurar mejor, muestra qué chunks se recuperaron para esta pregunta.
        if MODO_DEBUG:
            docs = obtener_docs_relevantes(vectorstore, pregunta, k=5)
            print("\n[debug] chunks recuperados:")
            for i, (doc, score_h, score_orig, hits) in enumerate(docs, 1):
                print(f"   {i}. score_hibrido={score_h:.4f} score_original={score_orig:.4f} hits={hits}")
                print(f"      {doc.page_content[:250].replace(chr(10), ' ')}...")

        with Spinner(f"{MODELO_LLM} pensando"):
            resultado = qa_chain.invoke({"question": pregunta})
        respuesta = resultado["answer"]

        print(f"\n{MODELO_LLM}: {respuesta}\n")


def main():
    if len(sys.argv) < 2:
        print(f"Uso: python3 {sys.argv[0]} <url>")
        sys.exit(1)

    url = sys.argv[1]

    html_crudo = descargar_html(url)
    html_limpio = extraer_contenido_principal(html_crudo, url)
    markdown = convertir_a_markdown(html_limpio)

    print(f"Markdown generado ({len(markdown)} caracteres).")

    ruta_md = guardar_markdown(markdown, url)
    print(f"Markdown guardado en: {ruta_md.resolve()}\n"
          f"(ábrelo para ver exactamente qué se está indexando)\n")

    chunks = dividir_en_chunks(markdown, url)
    vectorstore = crear_vectorstore(chunks)
    qa_chain = crear_cadena_qa(vectorstore)

    chat_interactivo(qa_chain, vectorstore)


if __name__ == "__main__":
    main()