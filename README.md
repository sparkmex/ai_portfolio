# AI Portfolio - DogName

Este repositorio contiene un script pequeño en Python que usa LangChain + Ollama para sugerir nombres de perro.

## Archivos

- `dog.py`: script principal que crea un cliente `ChatOllama` y genera sugerencias de nombres.
- `requirements.txt`: dependencias Python necesarias para ejecutar el script.

## Requisitos

- Python 3.11+ (se probó con Python 3.13)
- Ollama instalado y configurado
- Modelo Ollama descargado, por ejemplo `llama3.2:3b`

## Instalación

```bash
python -m pip install -r requirements.txt
```

## Uso

```bash
python dog.py
```

## Modelo Ollama

El script está configurado para usar `llama3.2:3b`. Si prefieres otro modelo instalado, actualiza la opción `model` dentro de `dog.py`.

## Notas

- Si no tienes el modelo, descárgalo con:
  ```bash
  ollama pull llama3.2:3b
  ```
- Para usar un modelo de chat más especializado, puedes cambiarlo a `llama3-chatqa:latest`.
