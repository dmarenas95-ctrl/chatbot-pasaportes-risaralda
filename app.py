import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

# ==================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==================================================

st.set_page_config(
    page_title="Chatbot Pasaportes Risaralda",
    page_icon="🛂",
    layout="wide"
)

# ==================================================
# LOGO Y ENCABEZADO
# ==================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("Logo_H_letra_negra.png", width=420)

st.markdown(
    """
    <h1 style='text-align:center;'>
    Chatbot de Pasaportes de Risaralda
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style='text-align:center;font-size:18px'>
    Asistente virtual para consultas sobre el trámite de pasaportes
    <br>
    <b>Gobernación de Risaralda</b>
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("Información")

    st.write(
        """
Este asistente responde preguntas relacionadas con:

- Expedición de pasaportes
- Requisitos
- Costos
- Citas
- Entrega
- Menores de edad
- Renovación
- Pérdida del pasaporte
        """
    )

    st.subheader("Preguntas sugeridas")

    st.write("• ¿Qué documentos necesito?")
    st.write("• ¿Cuánto cuesta el pasaporte?")
    st.write("• ¿Necesito cita?")
    st.write("• ¿Cómo solicito una cita?")
    st.write("• ¿Dónde queda la oficina?")
    st.write("• ¿Qué pasa si perdí mi pasaporte?")

    st.divider()

    if st.button("🗑 Nueva conversación"):

        st.session_state.messages = []

        st.rerun()

# ==================================================
# EMBEDDINGS
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# BASE VECTORIAL
# ==================================================

db = Chroma(
    persist_directory="base_vectorial",
    embedding_function=embeddings
)

# ==================================================
# HISTORIAL
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# ==================================================
# MOSTRAR MENSAJES
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ==================================================
# PREGUNTA DEL USUARIO
# ==================================================

pregunta = st.chat_input(
    "Escribe aquí tu pregunta sobre pasaportes..."
)

# ==================================================
# RESPUESTA
# ==================================================

if pregunta:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": pregunta
        }
    )

    with st.chat_message("user"):
        st.markdown(pregunta)

    # ==========================================
    # RESPUESTAS RÁPIDAS (SIN IA)
    # ==========================================

    texto = pregunta.lower().strip()

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
        "te lo agradezco",
        "muy amable"
    ]

    despedidas = [
        "adiós",
        "adios",
        "hasta luego",
        "nos vemos",
        "chao",
        "bye"
    ]

    if texto in saludos:

        respuesta_texto = """
👋 ¡Hola!

Bienvenido al Chatbot de la Oficina de Pasaportes de la Gobernación de Risaralda.

Estoy aquí para ayudarte con información sobre:

• Requisitos.
• Costos.
• Citas.
• Renovación.
• Pasaportes para menores.
• Horarios.
• Ubicación.

**¿Cómo puedo ayudarte hoy?**
"""

        with st.chat_message("assistant"):
            st.markdown(respuesta_texto)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": respuesta_texto
            }
        )

        st.stop()

    elif texto in agradecimientos:

        respuesta_texto = """
😊 ¡Con mucho gusto!

Si tienes otra consulta relacionada con el trámite de pasaportes de la Gobernación de Risaralda, estaré encantado de ayudarte.
"""

        with st.chat_message("assistant"):
            st.markdown(respuesta_texto)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": respuesta_texto
            }
        )

        st.stop()

    elif texto in despedidas:

        respuesta_texto = """
👋 ¡Hasta luego!

Gracias por utilizar el Chatbot de la Oficina de Pasaportes de la Gobernación de Risaralda.

Te deseo un excelente día.
"""

        with st.chat_message("assistant"):
            st.markdown(respuesta_texto)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": respuesta_texto
            }
        )

        st.stop()

    # Continúa con la búsqueda en la base vectorial
    with st.chat_message("assistant"):


        # --------------------------
        # BUSCAR CONTEXTO
        # --------------------------

        with st.spinner("🔍 Buscando información..."):

            resultados = db.similarity_search(
                pregunta,
                k=5
            )

        contexto = "\n\n".join(
            [doc.page_content for doc in resultados]
        )

        # --------------------------
        # PROMPT
        # --------------------------

        prompt = f"""
Eres el asistente virtual oficial de la Oficina de Pasaportes de la Gobernación de Risaralda.

Responde ÚNICAMENTE utilizando la información del contexto.

REGLAS:

1. No inventes información.
2. No supongas datos.
3. Si el contexto no contiene la respuesta responde exactamente:

"No encontré información sobre esa consulta en la base de conocimiento de Pasaportes de la Gobernación de Risaralda."

4. Si existen pasos, enuméralos.
5. Si existen valores o costos, preséntalos organizados.
6. Responde de manera cordial y profesional.

========================

CONTEXTO

{contexto}

========================

PREGUNTA

{pregunta}

========================

RESPUESTA
"""

        # --------------------------
        # GENERAR RESPUESTA
        # --------------------------

        with st.spinner("🤖 Generando respuesta..."):

            respuesta = ollama.chat(
                model="phi3:mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        respuesta_texto = respuesta["message"]["content"]

        st.markdown(respuesta_texto)

       

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta_texto
        }
    )

# ==================================================
# PIE DE PÁGINA
# ==================================================

st.divider()

st.caption(
    """
Proyecto de grado - Especialización en Inteligencia Artificial

Chatbot RAG para la consulta de información sobre pasaportes de la Gobernación de Risaralda.

"""
)
