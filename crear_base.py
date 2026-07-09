import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag_guardrails import crear_documentos_de_secciones


BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "documentos" / "faq_pasaportes.txt"
VECTOR_DIR = BASE_DIR / "base_vectorial"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def eliminar_base_vectorial():
    if not VECTOR_DIR.exists():
        print("No existia base vectorial previa.")
        return

    vector_dir_resuelto = VECTOR_DIR.resolve()
    base_dir_resuelto = BASE_DIR.resolve()

    if base_dir_resuelto not in vector_dir_resuelto.parents:
        raise RuntimeError(f"Ruta insegura para eliminar: {vector_dir_resuelto}")

    try:
        shutil.rmtree(vector_dir_resuelto)
    except PermissionError as error:
        raise RuntimeError(
            "No se pudo eliminar la base vectorial porque Chroma esta en uso. "
            "Cierra Streamlit, consolas o procesos de Python que esten usando "
            f"{vector_dir_resuelto} y vuelve a ejecutar crear_base.py."
        ) from error

    print(f"Base vectorial eliminada completamente: {vector_dir_resuelto}")


def cargar_documentos():
    print("Cargando FAQ...")
    print(f"Ruta exacta del archivo cargado: {FAQ_PATH.resolve()}")

    if not FAQ_PATH.exists():
        raise FileNotFoundError(f"No se encontro el archivo FAQ: {FAQ_PATH}")

    contenido = FAQ_PATH.read_text(encoding="utf-8")
    print(f"Caracteres cargados del documento: {len(contenido)}")

    if not contenido.strip():
        raise ValueError(f"El archivo FAQ esta vacio: {FAQ_PATH}")

    print(
        "Contiene 'REQUISITOS PARA MAYORES DE EDAD': "
        f"{'REQUISITOS PARA MAYORES DE EDAD' in contenido}"
    )

    secciones = crear_documentos_de_secciones(contenido)
    print(f"Secciones devueltas por crear_documentos_de_secciones: {len(secciones)}")

    if not secciones:
        raise ValueError("crear_documentos_de_secciones no devolvio secciones para indexar.")

    documents = []
    for seccion in secciones:
        contenido_seccion = seccion["contenido"].strip()
        if not contenido_seccion:
            continue

        documents.append(
            Document(
                page_content=contenido_seccion,
                metadata=seccion["metadata"],
            )
        )

    if not documents:
        raise ValueError("No se genero ningun Document valido para indexar.")

    print(f"Documentos langchain generados: {len(documents)}")
    print("Primeros 3 chunks generados:")
    for indice, document in enumerate(documents[:3], start=1):
        print("=" * 70)
        print(f"CHUNK {indice} | Titulo: {document.metadata.get('titulo')}")
        print(document.page_content[:1000])

    return documents


def crear_base(documents):
    print("Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Embeddings creados correctamente.")

    eliminar_base_vectorial()

    print("Creando nueva base vectorial...")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
    )

    try:
        db.persist()
        print("Persistencia explicita ejecutada con db.persist().")
    except AttributeError:
        print("db.persist() no existe en esta version de Chroma; persiste automaticamente.")

    count = db._collection.count()
    print(f"Documentos guardados en Chroma: {count}")

    if count == 0:
        raise RuntimeError("La base vectorial quedo vacia. No se indexo ningun documento.")

    return db


def probar_busqueda(db, pregunta, titulo_esperado):
    print(f"Prueba manual de busqueda: {pregunta}")
    print(f"Seccion esperada: {titulo_esperado}")
    resultados = db.similarity_search_with_score(pregunta, k=5)
    contiene_titulo_esperado = False

    for indice, (document, score) in enumerate(resultados, start=1):
        titulo = document.metadata.get("titulo", "SIN TITULO")
        contiene_titulo_esperado = (
            contiene_titulo_esperado
            or titulo_esperado in titulo
            or titulo_esperado in document.page_content
        )

        print("=" * 70)
        print(f"RESULTADO {indice} | score/distancia: {score} | titulo: {titulo}")
        print(document.page_content[:1200])

    print(f"Recupera '{titulo_esperado}': {contiene_titulo_esperado}")

    if not contiene_titulo_esperado:
        print(
            "ADVERTENCIA: la busqueda semantica no recupero la seccion esperada. "
            "La base vectorial fue creada y contiene documentos; la app puede "
            "usar la capa de intencion/palabras clave como recuperacion adicional. "
            f"Seccion esperada: {titulo_esperado}"
        )


def main():
    documents = cargar_documentos()
    db = crear_base(documents)

    probar_busqueda(
        db,
        "¿Cuál es el costo del pasaporte?",
        "COSTOS Y FORMAS DE PAGO DEL PASAPORTE",
    )
    probar_busqueda(
        db,
        "¿Qué documentos necesito?",
        "REQUISITOS PARA MAYORES DE EDAD",
    )

    print("Base vectorial creada y validada correctamente.")


if __name__ == "__main__":
    main()
