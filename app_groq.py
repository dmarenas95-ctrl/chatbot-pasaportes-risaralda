import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================================
# CARGAR VARIABLES DE ENTORNO
# ==========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "documentos" / "faq_pasaportes.txt"
VECTOR_DIR = BASE_DIR / "base_vectorial"
CHROMA_DB_PATH = VECTOR_DIR / "chroma.sqlite3"

MENSAJE_SIN_INFORMACION = (
    "No encontré información sobre esa consulta en la base de conocimiento "
    "de la Oficina de Pasaportes de la Gobernación de Risaralda."
)

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

if not GROQ_API_KEY:
    st.error(
        "Falta configurar GROQ_API_KEY en el archivo .env. "
        "Agrega la clave y vuelve a ejecutar la aplicación."
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

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

if (
    not CHROMA_DB_PATH.exists()
    or FAQ_PATH.stat().st_mtime > CHROMA_DB_PATH.stat().st_mtime
):
    with st.spinner("Actualizando base de conocimiento..."):
        proceso = subprocess.run(
            [sys.executable, str(BASE_DIR / "crear_base.py")],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )

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

if pregunta:

    pregunta_lower = pregunta.lower()

    if any(x in pregunta_lower for x in [
        "descuento",
        "descuentos",
        "beneficio",
        "beneficios",
        "exención",
        "exenciones"
    ]):
        respuesta = (
            "La Oficina de Pasaportes puede aplicar exenciones o beneficios "
            "únicamente en los casos establecidos por la normatividad vigente. Debe consultar directamente en la oficina de Pasaportes - Gobernacion de Risaralda"
        )

        with st.chat_message("assistant"):
            st.markdown(respuesta)

        st.session_state.messages.append(
            {"role": "assistant", "content": respuesta}
        )

        st.stop()
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
                resultados = db.similarity_search_with_score(
                    pregunta,
                    k=3
                )
        except Exception as error:
            respuesta_texto = (
                "No pude consultar la base de conocimiento en este momento. "
                "Verifica que la base vectorial exista y que las dependencias estén instaladas."
            )
            st.error(respuesta_texto)
            st.exception(error)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": respuesta_texto
                }
            )
            st.stop()

        if len(resultados) == 0:

            respuesta_texto = MENSAJE_SIN_INFORMACION

            st.markdown(respuesta_texto)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": respuesta_texto
                }
            )

            st.stop()

        # ==========================================================
        # CONTEXTO
        # ==========================================================

        documentos = [documento for documento, _score in resultados]

        contexto = "\n\n".join(
            documento.page_content for documento in documentos
        )

        # ==========================================================
        # PROMPT DEL MODELO
        # ==========================================================

        prompt = f"""
Eres el asistente virtual oficial de la Oficina de Pasaportes de la Gobernación de Risaralda.

Tu única fuente de información es el CONTEXTO.

REGLAS IMPORTANTES:

1. Nunca inventes información.

2. Nunca respondas utilizando conocimientos propios.

3. Si la respuesta no aparece claramente en el contexto responde exactamente:

"No encontré información sobre esa consulta en la base de conocimiento de la Oficina de Pasaportes de la Gobernación de Risaralda."

4. Si existen listas o pasos, organízalos utilizando viñetas.

5. Si existen valores económicos, organízalos correctamente.

6. No menciones el contexto.

7. No digas frases como:
"Según el contexto..."
"Con base en el contexto..."

Simplemente responde.

==============================

CONTEXTO

{contexto}

==============================

PREGUNTA

{pregunta}

==============================

RESPUESTA
"""

        # ==========================================================
        # CONSULTAR GROQ
        # ==========================================================

        try:
            with st.spinner("🤖 Escribiendo..."):
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2,
                    max_tokens=700
                )

            respuesta_texto = completion.choices[0].message.content.strip()
        except Exception as error:
            respuesta_texto = (
                "No pude generar la respuesta con Groq en este momento. "
                "Revisa la clave GROQ_API_KEY, la conexión y el modelo configurado."
            )
            st.error(respuesta_texto)
            st.exception(error)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": respuesta_texto
                }
            )
            st.stop()

        st.markdown(respuesta_texto)

    # Guardar respuesta en el historial
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta_texto
        }
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
        
