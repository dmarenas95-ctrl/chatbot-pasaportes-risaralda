from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "documentos" / "faq_pasaportes.txt"
VECTOR_DIR = BASE_DIR / "base_vectorial"

print("Cargando FAQ...")

if not FAQ_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo FAQ: {FAQ_PATH}")

# Leer el archivo completo
with open(
    FAQ_PATH,
    "r",
    encoding="utf-8"
) as archivo:

    contenido = archivo.read()

# Separar cada pregunta
bloques = contenido.split(
    "------------------------------------------------------------"
)

documents = []

for bloque in bloques:

    bloque = bloque.strip()

    if bloque:

        documents.append(
            Document(page_content=bloque)
        )

if not documents:
    raise ValueError("El archivo FAQ no contiene preguntas válidas para indexar.")

print(f"Se encontraron {len(documents)} preguntas.")

print("Cargando modelo de embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Eliminar la base anterior
if VECTOR_DIR.exists():

    shutil.rmtree(VECTOR_DIR)

    print("Base vectorial eliminada.")

print("Creando nueva base vectorial...")

Chroma.from_documents(
    documents,
    embeddings,
    persist_directory=str(VECTOR_DIR)
)

print("Base vectorial creada correctamente.")
