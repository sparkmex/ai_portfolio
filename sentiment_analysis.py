"""
Analizador de sentimientos usando DeepSeek (Ollama, API compatible con OpenAI).

Requisitos:
    pip install openai pandas colorama

Uso:
    python3 analizar_sentimientos.py sentimientos.csv
"""

import sys
import csv
import json
from collections import Counter

from openai import OpenAI
from colorama import init, Fore, Style

init(autoreset=True)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODELO = "deepseek-llm:7b-chat"

PROMPT_SISTEMA = (
    "Eres un clasificador de sentimientos. "
    "Dado un texto, responde ÚNICAMENTE con una de estas tres palabras, "
    "sin explicaciones ni puntuación adicional: positivo, negativo o neutral."
)


def clasificar_sentimiento(texto: str) -> str:
    """Envía un texto a DeepSeek y devuelve 'positivo', 'negativo' o 'neutral'."""
    try:
        response = client.chat.completions.create(
            model=MODELO,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": texto}
            ],
            temperature=0.0
        )
        resultado = response.choices[0].message.content.strip().lower()

        # Normalizar por si el modelo agrega texto extra
        if "positiv" in resultado:
            return "positivo"
        elif "negativ" in resultado:
            return "negativo"
        else:
            return "neutral"

    except Exception as e:
        print(f"{Fore.RED}Error clasificando texto: {e}{Style.RESET_ALL}")
        return "neutral"


LONGITUD_MINIMA = 4  # textos con longitud <= a esto se descartan


def leer_csv(ruta: str) -> tuple:
    """Lee el CSV y devuelve (textos_validos, descartados).

    Se descartan los textos cuya longitud sea <= LONGITUD_MINIMA.
    """
    textos = []
    descartados = 0
    with open(ruta, "r", encoding="utf-8") as f:
        lector = csv.reader(f)
        for fila in lector:
            if not fila:
                continue
            texto = fila[0].strip()
            if not texto:
                continue
            if len(texto) <= LONGITUD_MINIMA:
                descartados += 1
                continue
            textos.append(texto)
    return textos, descartados


def mostrar_resumen(resultados: Counter, total: int):
    """Imprime el resumen con colores y porcentajes."""
    positivos = resultados.get("positivo", 0)
    negativos = resultados.get("negativo", 0)
    neutrales = resultados.get("neutral", 0)

    pct_pos = (positivos / total * 100) if total else 0
    pct_neg = (negativos / total * 100) if total else 0
    pct_neu = (neutrales / total * 100) if total else 0

    print("\n" + "=" * 50)
    print(f"{Style.BRIGHT}RESUMEN DE ANÁLISIS DE SENTIMIENTOS{Style.RESET_ALL}")
    print("=" * 50)
    print(f"Total de entradas analizadas: {total}\n")

    print(f"{Fore.GREEN}{Style.BRIGHT}Positivas:{Style.RESET_ALL} "
          f"{Fore.GREEN}{positivos} entradas ({pct_pos:.1f}%){Style.RESET_ALL}")

    print(f"{Fore.RED}{Style.BRIGHT}Negativas:{Style.RESET_ALL} "
          f"{Fore.RED}{negativos} entradas ({pct_neg:.1f}%){Style.RESET_ALL}")

    print(f"{Fore.YELLOW}{Style.BRIGHT}Neutrales:{Style.RESET_ALL} "
          f"{Fore.YELLOW}{neutrales} entradas ({pct_neu:.1f}%){Style.RESET_ALL}")

    print("=" * 50 + "\n")


def main():
    if len(sys.argv) < 2:
        print(f"Uso: python3 {sys.argv[0]} <ruta_al_csv>")
        sys.exit(1)

    ruta_csv = sys.argv[1]
    textos, descartados = leer_csv(ruta_csv)

    if not textos:
        print(f"{Fore.RED}No se encontraron textos válidos en el archivo.{Style.RESET_ALL}")
        sys.exit(1)

    print(f"Analizando {len(textos)} entradas con {MODELO}...")
    if descartados:
        print(f"{Fore.YELLOW}Se descartaron {descartados} entradas con longitud <= {LONGITUD_MINIMA} caracteres.{Style.RESET_ALL}")
    print()

    resultados = Counter()

    for i, texto in enumerate(textos, 1):
        sentimiento = clasificar_sentimiento(texto)
        resultados[sentimiento] += 1

        color = {
            "positivo": Fore.GREEN,
            "negativo": Fore.RED,
            "neutral": Fore.YELLOW
        }[sentimiento]

        print(f"[{i}/{len(textos)}] {color}{sentimiento.upper()}{Style.RESET_ALL} → {texto[:60]}")

    mostrar_resumen(resultados, len(textos))


if __name__ == "__main__":
    main()
