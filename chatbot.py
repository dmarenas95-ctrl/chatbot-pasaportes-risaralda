import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

from rag_guardrails import (
    MENSAJE_SIN_INFORMACION,
    construir_contexto,
    construir_mensajes_groq,
    dividir_documento,
    seleccionar_documentos_relevantes,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError(
        "Falta configurar GROQ_API_KEY. Crea un archivo .env en la raíz del "
        "proyecto con GROQ_API_KEY=gsk_tu_api_key_aqui y GROQ_MODEL=llama-3.1-8b-instant."
    )

print("API key de Groq encontrada. No se muestra el valor por seguridad.")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Cargar base vectorial
db = Chroma(
    persist_directory="base_vectorial",
    embedding_function=embeddings
)

with open("documentos/faq_pasaportes.txt", "r", encoding="utf-8") as archivo:
    secciones_faq = dividir_documento(archivo.read())

print("Chatbot RAG iniciado")
print("Escribe 'salir' para terminar\n")

while True:

    pregunta = input("Tú: ")

    if pregunta.lower() == "salir":
        break

    # Buscar contexto relevante sin permitir respuestas desde conocimiento externo.
    resultados = db.similarity_search_with_score(pregunta, k=5)

    documentos = seleccionar_documentos_relevantes(
        resultados,
        pregunta,
        umbral_distancia=1.8,
        secciones=secciones_faq,
    )

    if len(documentos) == 0:
        print("\nChatbot:")
        print(MENSAJE_SIN_INFORMACION)
        print("\n")
        continue

    contexto = construir_contexto(documentos)
    mensajes = construir_mensajes_groq(contexto, pregunta)

    respuesta = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=mensajes,
        temperature=0,
        max_tokens=700,
    )

    print("\nChatbot:")
    print(respuesta.choices[0].message.content.strip())
    print("\n")
