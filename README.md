# SimpleChat

Este proyecto reúne varios ejemplos para trabajar con modelos locales usando **Ollama** y APIs compatibles con OpenAI.

## Archivos principales

- `chat.py`  
  Chat básico con el modelo `deepseek-llm:7b-chat` usando la API de Ollama.

- `retrieve_info.py`  
  Script que descarga una página web, la divide en fragmentos y permite hacer preguntas usando RAG.

- `retrieve_info_md.py`  
  Versión mejorada que convierte el contenido a Markdown antes de indexarlo, útil para páginas con mucho ruido HTML.

- `sentiment_analysis.py`  
  Script para analizar el sentimiento de un texto.

- `steps.txt`  
  Pasos rápidos de configuración.

---

## Requisitos previos

1. Instala [Ollama](https://ollama.com/).
2. Descarga los modelos que vas a usar:

```bash
ollama pull deepseek-llm:7b-chat
ollama pull mistral:7b-instruct
ollama pull nomic-embed-text
```

3. Asegúrate de tener Python 3 instalado.

---

## Configuración del entorno virtual

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Instalar todas las dependencias del archivo `requirements.txt`

Una vez activado el entorno virtual, ejecuta:

```bash
pip install -r requirements.txt
```

Este comando instala todas las librerías listadas en el archivo de dependencias.

### Si quieres generar el archivo `requirements.txt`

Si aún no lo tienes, puedes crear uno con:

```bash
pip freeze > requirements.txt
```

### Verificar la instalación

Puedes comprobar que las librerías quedaron instaladas con:

```bash
pip list
```

---

## Cómo ejecutar `chat.py`

```bash
python chat.py
```

Este script:
- se conecta a `http://localhost:11434/v1`
- usa la API de Ollama como si fuera OpenAI
- te permite chatear desde la terminal

---

## Cómo ejecutar `retrieve_info.py`

```bash
python retrieve_info.py https://ejemplo.com
```

Ejemplo:

```bash
python retrieve_info.py https://docs.python.org/3/
```

---

## Cómo ejecutar `retrieve_info_md.py`

```bash
python retrieve_info_md.py https://ejemplo.com
```

Esta versión es útil cuando quieres extraer contenido limpio de una página y hacer preguntas sobre el texto real visible.

---

## Cómo ejecutar `sentiment_analysis.py`

```bash
python sentiment_analysis.py
```

---

## Notas importantes

- Asegúrate de que Ollama esté corriendo antes de ejecutar los scripts.
- Si el modelo tarda en responder, puede ser normal dependiendo de tu computadora.
- Para preguntas sobre páginas web, `retrieve_info_md.py` suele dar mejores resultados cuando hay mucho ruido HTML.

---

## Dependencias sugeridas

```bash
pip install openai langchain langchain-community langchain-ollama chromadb beautifulsoup4 requests trafilatura markdownify
```
