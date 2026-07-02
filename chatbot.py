from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Cargar base vectorial
db = Chroma(
    persist_directory="base_vectorial",
    embedding_function=embeddings
)

print("Chatbot RAG iniciado")
print("Escribe 'salir' para terminar\n")

while True:

    pregunta = input("Tú: ")

    if pregunta.lower() == "salir":
        break

    # Buscar contexto
    resultados = db.similarity_search(pregunta, k=3)

    contexto = "\n".join([doc.page_content for doc in resultados])

    prompt = f"""
    Responde únicamente con esta información:

    {contexto}

    Pregunta:
    {pregunta}
    """

    respuesta = ollama.chat(
    model="phi3:mini",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "num_predict": 300
    }
)

    print("\nChatbot:")
    print(respuesta["message"]["content"])
    print("\n")
