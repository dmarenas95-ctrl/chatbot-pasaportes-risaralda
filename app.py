import os
import subprocess
import sys
import gc
from pathlib import Path

import streamlit as st
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

# ==========================================================
# CARGAR VARIABLES DE ENTORNO
# ==========================================================

load_dotenv()

ZAI_API_KEY = os.getenv("ZAI_API_KEY")
ZAI_MODEL = os.getenv("ZAI_MODEL", "glm-4.7-flash")
BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "documentos" / "faq_pasaportes.txt"
VECTOR_DIR = BASE_DIR / "base_vectorial"
CHROMA_DB_PATH = VECTOR_DIR / "chroma.sqlite3"
INDEX_SOURCE_PATHS = [
    FAQ_PATH,
    BASE_DIR / "crear_base.py",
    BASE_DIR / "rag_guardrails.py",
]

# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="Asistente Virtual de Pasaportes",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# ESTILOS CSS
# ==========================================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

h1{
    color:#003087;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOGO
# ==========================================================

col1, col2, col3 = st.columns([1,2,1])

with col2:

    st.image(
        "Logo_H_letra_negra.png",
        width=420
    )

# ==========================================================
# ENCABEZADO
# ==========================================================

st.markdown("""
<h1 style='text-align:center'>
Chatbot de Pasaportes de Risaralda
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center;font-size:18px'>
<b>Gobernación de Risaralda</b>


</p>
""", unsafe_allow_html=True)

st.divider()

if not ZAI_API_KEY:
    st.error(
        "Falta configurar ZAI_API_KEY. Crea un archivo .env en la raíz del "
        "proyecto con ZAI_API_KEY=gsk_tu_api_key_aqui y ZAI_MODEL=glm-4.7-flash."
    )
    st.stop()

client = OpenAI(
    api_key=os.getenv("ZAI_API_KEY"),
    base_url="https://api.z.ai/api/paas/v4"
)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("📌 Información")

    st.write("""
Este asistente puede ayudarte con información sobre:

- 🛂 Expedición de pasaportes
- 💲 Costos
- 📅 Agendamiento de citas
- 👶 Pasaportes para menores de edad
- 🔄 Renovación
- 📍 Ubicación de la oficina
- 🕒 Horarios de atención
""")

    st.subheader("Preguntas sugeridas")

    st.write("• ¿Cuál es el costo del pasaporte?")
    st.write("• ¿Cómo puedo solicitar una cita para tramitar el pasaporte?")
    st.write("• ¿Qué documentos debo presentar?")
    st.write("• ¿Dónde puedo tramitar el pasaporte en Risaralda?")
    st.write("• ¿Cuál es el horario de atención?")
    st.write("• ¿Cómo renuevo mi pasaporte?")

    st.divider()

    if st.button("🗑 Nueva conversación"):

        st.session_state.messages = []

        st.rerun()
# ==========================================
# ACTUALIZAR BASE VECTORIAL SI EL FAQ CAMBIÓ
# ==========================================

if not FAQ_PATH.exists():
    st.error(f"No se encontró el archivo de conocimiento: {FAQ_PATH}")
    st.stop()

@st.cache_data
def cargar_secciones_faq(faq_mtime):
    with open(FAQ_PATH, "r", encoding="utf-8") as archivo:
        contenido = archivo.read()

    secciones = dividir_documento(contenido)
    print("[RAG] Ruta exacta del archivo cargado:", FAQ_PATH)
    print("[RAG] Caracteres cargados del documento:", len(contenido))
    print(
        "[RAG] Contiene 'REQUISITOS PARA MAYORES DE EDAD':",
        "REQUISITOS PARA MAYORES DE EDAD" in contenido,
    )
    print("[RAG] Chunks/secciones generadas desde FAQ:", len(secciones))
    for indice, seccion in enumerate(secciones[:3], start=1):
        print("=" * 70)
        print(f"[RAG] CHUNK {indice}")
        print(seccion[:1000])

    return secciones

secciones_faq = cargar_secciones_faq(FAQ_PATH.stat().st_mtime)
base_referencia_mtime = max(path.stat().st_mtime for path in INDEX_SOURCE_PATHS if path.exists())

if (
    not CHROMA_DB_PATH.exists()
    or base_referencia_mtime > CHROMA_DB_PATH.stat().st_mtime
):
    with st.spinner("Actualizando base de conocimiento..."):
        st.cache_resource.clear()
        gc.collect()

        proceso = subprocess.run(
            [sys.executable, str(BASE_DIR / "crear_base.py")],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )

    if proceso.stdout:
        print(proceso.stdout)
    if proceso.stderr:
        print(proceso.stderr)

    if proceso.returncode != 0:
        st.error("No se pudo actualizar la base de conocimiento.")
        st.code(proceso.stderr or proceso.stdout)
        st.stop()

base_actualizada_en = CHROMA_DB_PATH.stat().st_mtime if CHROMA_DB_PATH.exists() else 0

# ==========================================================
# CARGAR EMBEDDINGS
# ==========================================================

@st.cache_resource

def cargar_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = cargar_embeddings()

# ==========================================================
# CARGAR BASE VECTORIAL
# ==========================================================

@st.cache_resource

def cargar_base(base_actualizada_en):

    return Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings
    )

db = cargar_base(base_actualizada_en)
try:
    print("[RAG] Documentos guardados en Chroma:", db._collection.count())
except Exception as error:
    print("[RAG] No se pudo contar documentos en Chroma:", error)
# ==========================================================
# HISTORIAL DE LA CONVERSACIÓN
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================================
# MENSAJE DE BIENVENIDA
# ==========================================================

if len(st.session_state.messages) == 0:

    bienvenida = """
👋 Soy el Asistente Virtual de la Oficina de Pasaportes.

¿En qué puedo ayudarte hoy?
"""

    st.chat_message("assistant").markdown(bienvenida)

# ==========================================================
# MOSTRAR HISTORIAL
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ==========================================================
# CAJA DE TEXTO
# ==========================================================

pregunta = st.chat_input("Escribe aquí tu pregunta...")

# ==========================================================
# SI EL USUARIO ESCRIBE
# ==========================================================

if pregunta:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":pregunta
        }
    )

    with st.chat_message("user"):

        st.markdown(pregunta)

    texto = pregunta.lower().strip()

    # ======================================================
    # SALUDOS
    # ======================================================

    saludos = [
        "hola",
        "buenas",
        "buenos días",
        "buenos dias",
        "buen día",
        "buen dia",
        "buenas tardes",
        "buenas noches",
        "hey"
    ]

    agradecimientos = [
        "gracias",
        "muchas gracias",
        "mil gracias",
        "te lo agradezco"
    ]

    despedidas = [
        "adiós",
        "adios",
        "hasta luego",
        "nos vemos",
        "bye",
        "chao"
    ]

    # ======================================================
    # RESPUESTAS RÁPIDAS
    # ======================================================

    if texto in saludos:

        respuesta = """
👋 ¡Hola!

Bienvenido al Chatbot de la Oficina de Pasaportes de la Gobernación de Risaralda.

Estoy disponible para ayudarte con información sobre:

• Expedición de pasaportes

• Costos

• Requisitos

• Horarios

• Menores de edad

• Renovación

**¿Cómo puedo ayudarte?**
"""

        with st.chat_message("assistant"):

            st.markdown(respuesta)

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":respuesta
            }
        )

        st.stop()

    elif texto in agradecimientos:

        respuesta = """
😊 ¡Con mucho gusto!

Si tienes otra consulta sobre el trámite de pasaportes, estaré encantado de ayudarte.
"""

        with st.chat_message("assistant"):

            st.markdown(respuesta)

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":respuesta
            }
        )

        st.stop()

    elif texto in despedidas:
        respuesta = """
    👋 ¡Hasta luego!

    Gracias por utilizar el Chatbot de Pasaportes de la Gobernación de Risaralda.

    Te deseo un excelente día.
    """

        with st.chat_message("assistant"):

            st.markdown(respuesta)

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":respuesta
            }
        )

        st.stop()
    
    # ==========================================================
    # BÚSQUEDA SEMÁNTICA (RAG)
    # ==========================================================

    with st.chat_message("assistant"):

        try:
            with st.spinner("🔍 Buscando información..."):
                print("=" * 70)
                print("[RAG] Pregunta del usuario:", pregunta)
                # Recuperamos varios candidatos y luego aplicamos un filtro
                # estricto para no responder con fragmentos poco relacionados.
                resultados = db.similarity_search_with_score(
                    pregunta,
                    k=5
                )
                contiene_requisitos_chroma = False
                for indice, (documento, score) in enumerate(resultados, start=1):
                    titulo = documento.metadata.get("titulo", "SIN TITULO")
                    contiene_requisitos_chroma = (
                        contiene_requisitos_chroma
                        or "REQUISITOS PARA MAYORES DE EDAD" in documento.page_content
                    )
                    print("-" * 70)
                    print(f"[RAG] Chroma resultado {indice} | score/distancia: {score} | título: {titulo}")
                    print(documento.page_content[:1200])
                print(
                    "[RAG] Chroma recuperó 'REQUISITOS PARA MAYORES DE EDAD':",
                    contiene_requisitos_chroma,
                )
        except Exception as error:
            respuesta_texto = (
                "No pude consultar la base de conocimiento en este momento. "
                "Verifica que la base vectorial exista y que las dependencias estén instaladas."
            )
            st.error(respuesta_texto)
            st.exception(error)
            st.session_state.messages.append(
                {"role": "assistant", "content": respuesta_texto}
            )
            st.stop()

        documentos = seleccionar_documentos_relevantes(
            resultados,
            pregunta,
            umbral_distancia=1.8,
            secciones=secciones_faq,
        )
        print("[RAG] Documentos seleccionados para contexto:", len(documentos))
        for indice, documento in enumerate(documentos, start=1):
            print(
                f"[RAG] Contexto seleccionado {indice} | "
                f"título: {documento.metadata.get('titulo', 'SIN TITULO')} | "
                f"origen: {documento.metadata.get('origen', 'chroma')}"
            )

        if len(documentos) == 0:
            respuesta_texto = MENSAJE_SIN_INFORMACION
            st.markdown(respuesta_texto)
            st.session_state.messages.append(
                {"role": "assistant", "content": respuesta_texto}
            )
            st.stop()

        # ==========================================================
        # CONTEXTO
        # ==========================================================

        contexto = construir_contexto(documentos)
        print(
            "[RAG] Contexto final contiene 'REQUISITOS PARA MAYORES DE EDAD':",
            "REQUISITOS PARA MAYORES DE EDAD" in contexto,
        )
        print("[RAG] Contexto final enviado al modelo:")
        print(contexto[:4000])

        # Puedes dejar el DEBUG si lo deseas
        # st.subheader("DEBUG")
        # st.code(contexto)

        # ==========================================================
        # PROMPT DEL MODELO (Separado en Sistema y Usuario)
        # ==========================================================

        mensajes = construir_mensajes_groq(contexto, pregunta)

        # ==========================================================
        # CONSULTAR GROQ
        # ==========================================================

        try:
            with st.spinner("🤖 Escribiendo..."):
                completion = client.chat.completions.create(
                    model=ZAI_MODEL,
                    messages=mensajes,
                    temperature=0.0,
                    max_tokens=2000
                )

            mensaje = completion.choices[0].message

            contenido = mensaje.content
            razonamiento = getattr(mensaje, "reasoning_content", None)

            print("=" * 80)
            print("RESPUESTA COMPLETA DEL MODELO:")
            print(completion)
            print("=" * 80)

            print("CONTENT:")
            print(repr(contenido))

            print("REASONING:")
            print(repr(razonamiento))

            if contenido is None or contenido.strip() == "":
                respuesta_texto = (
                    "El modelo no generó una respuesta final. "
                    "Revisa el razonamiento mostrado en consola."
                )
            else:
                respuesta_texto = contenido.strip()

            print("[RAG] Respuesta generada:")
            print(respuesta_texto)
        except Exception as error:
            respuesta_texto = (
                "No pude generar la respuesta con Groq en este momento. "
                "Revisa la clave ZAI_API_KEY, la conexión y el modelo configurado en GROQ_MODEL."
            )
            st.error(respuesta_texto)
            st.exception(error)
            st.session_state.messages.append(
                {"role": "assistant", "content": respuesta_texto}
            )
            st.stop()

        st.markdown(respuesta_texto)
        st.session_state.messages.append(
            {"role": "assistant", "content": respuesta_texto}
        )
# ==========================================================
# PIE DE PÁGINA
# ==========================================================

st.divider()

st.caption(
    """
Proyecto de grado - Especialización en Inteligencia Artificial

Chatbot RAG para consultas sobre la Oficina de Pasaportes
de la Gobernación de Risaralda.
"""
)
        
