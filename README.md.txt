# Chatbot RAG - Pasaportes Gobernacion de Risaralda

Proyecto de grado de la Especializacion en Inteligencia Artificial.

## Tecnologias

- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings
- API de Groq mediante cliente compatible con OpenAI

## Configuracion

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Crea un archivo `.env` en la raiz del proyecto con:

```env
GROQ_API_KEY=gsk_tu_api_key_aqui
GROQ_MODEL=llama-3.1-8b-instant
```

`GROQ_MODEL` es opcional. Si no se configura, la aplicacion usa `llama-3.1-8b-instant`.
No subas tu archivo `.env` a GitHub.

## Ejecutar

```bash
streamlit run app.py
```

La aplicacion carga `documentos/faq_pasaportes.txt`, recupera el contexto relevante con RAG y envia solo ese contexto junto con la pregunta a Groq.
