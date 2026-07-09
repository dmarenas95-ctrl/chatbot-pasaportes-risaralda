import re
import unicodedata
from types import SimpleNamespace


MENSAJE_SIN_INFORMACION = "No encontré información sobre esa consulta en la base de conocimiento de la Oficina de Pasaportes de la Gobernación de Risaralda. Para recibir orientación precisa, te invitamos a acercarte directamente a la Oficina de Pasaportes, donde nuestro personal podrá brindarte la información correspondiente de acuerdo con la normatividad vigente."

SYSTEM_PROMPT = f"""
Eres un asistente especializado en responder preguntas sobre el trámite del pasaporte en Colombia, pero SOLO puedes usar la información incluida en el CONTEXTO proporcionado.

Reglas obligatorias:

1. No uses conocimiento externo.
2. No inventes requisitos, pasos, precios, horarios, enlaces, entidades ni fechas.
3. Si la respuesta no está explícitamente en el CONTEXTO, responde exactamente: "{MENSAJE_SIN_INFORMACION}"
4. Si la pregunta es ambigua, pide una aclaración breve.
5. Responde de forma clara, breve y ordenada.
6. No contradigas el documento.
7. No completes información faltante con suposiciones.
8. No menciones que usaste un contexto ni una base de conocimiento, salvo para dar la frase obligatoria de falta de información.
""".strip()

SEPARADOR_BLOQUES = "======================================================================"

STOPWORDS = {
    "a",
    "al",
    "ante",
    "asi",
    "bajo",
    "cada",
    "como",
    "con",
    "cual",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "el",
    "en",
    "entre",
    "es",
    "esa",
    "ese",
    "esta",
    "este",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mi",
    "mis",
    "no",
    "o",
    "para",
    "por",
    "que",
    "se",
    "si",
    "su",
    "sus",
    "un",
    "una",
    "y",
    "yo",
}

INTENCIONES = {
    "costos": {
        "palabras": [
            "costo",
            "costos",
            "precio",
            "precios",
            "valor",
            "valores",
            "cuesta",
            "vale",
            "tarifa",
            "tarifas",
            "pagar",
            "pago",
            "pagos",
            "descuento",
            "descuentos",
            "rebaja",
            "beneficio",
            "beneficios",
            "exencion",
            "exenciones",
            "exoneracion",
            "exoneraciones",
            "subsidio",
            "subsidios",
            "tarifa especial",
            "menor valor",
        ],
        "secciones": [
            "COSTOS Y FORMAS DE PAGO DEL PASAPORTE",
            "CASOS ESPECIALES",
        ],
    },
    "requisitos": {
        "palabras": [
            "documentos",
            "documento",
            "documentacion",
            "papeles",
            "requisitos",
            "requisito",
            "necesito",
            "llevar",
            "presentar",
            "piden",
            "cedula",
            "foto",
            "fotografia",
            "fotocopias",
        ],
        "secciones": ["REQUISITOS PARA MAYORES DE EDAD"],
    },
    "citas": {
        "palabras": [
            "cita",
            "citas",
            "agendar",
            "agenda",
            "agendo",
            "programar",
            "reservar",
            "reserva",
            "disponibilidad",
        ],
        "secciones": ["AGENDAMIENTO DE CITAS"],
    },
    "ubicacion_horarios": {
        "palabras": [
            "donde",
            "ubicacion",
            "direccion",
            "queda",
            "oficina",
            "horario",
            "horarios",
            "atienden",
            "sabado",
            "viernes",
        ],
        "secciones": ["HORARIOS, UBICACIÓN Y CANALES DE ATENCIÓN"],
    },
    "menores": {
        "palabras": [
            "menor",
            "menores",
            "niño",
            "nino",
            "niña",
            "nina",
            "hijo",
            "hija",
            "bebe",
            "registro civil",
            "tarjeta de identidad",
        ],
        "secciones": ["PASAPORTE PARA MENORES DE EDAD"],
    },
    "renovacion": {
        "palabras": [
            "renovar",
            "renovacion",
            "vencido",
            "vencimiento",
            "perdi",
            "perdida",
            "robo",
            "hurto",
            "extraviado",
            "dañado",
            "danado",
            "deteriorado",
        ],
        "secciones": ["RENOVACIÓN, PÉRDIDA Y HURTO DEL PASAPORTE"],
    },
    "entrega": {
        "palabras": [
            "entrega",
            "reclamar",
            "recoger",
            "listo",
            "demora",
            "tiempo",
            "retiro",
        ],
        "secciones": ["ENTREGA DEL PASAPORTE"],
    },
}


def dividir_documento(contenido):
    """Divide el FAQ en bloques de conocimiento indexables."""
    partes = [
        bloque.strip()
        for bloque in contenido.split(SEPARADOR_BLOQUES)
        if bloque.strip()
    ]
    bloques = []
    indice = 0

    while indice < len(partes):
        parte = partes[indice]
        parte_normalizada = normalizar_texto(parte)
        es_encabezado = (
            parte_normalizada.startswith("tema")
            or parte_normalizada.startswith("descripcion general")
        )

        if es_encabezado and indice + 1 < len(partes):
            siguiente = partes[indice + 1]
            siguiente_normalizado = normalizar_texto(siguiente)
            siguiente_es_encabezado = (
                siguiente_normalizado.startswith("tema")
                or siguiente_normalizado.startswith("descripcion general")
            )

            if not siguiente_es_encabezado:
                bloques.append(f"{parte}\n\n{siguiente}")
                indice += 2
                continue

        bloques.append(parte)
        indice += 1

    return bloques


def extraer_titulo_seccion(bloque):
    for linea in bloque.splitlines():
        linea = linea.strip()
        if linea.upper().startswith("TEMA:"):
            return linea.replace("TEMA:", "", 1).strip()
    return "DESCRIPCIÓN GENERAL"


def crear_documentos_de_secciones(contenido):
    """Mantiene cada sección completa para no perder título, palabras y observaciones."""
    documentos = []
    for indice, bloque in enumerate(dividir_documento(contenido), start=1):
        documentos.append(
            {
                "contenido": bloque,
                "metadata": {
                    "chunk": indice,
                    "titulo": extraer_titulo_seccion(bloque),
                },
            }
        )
    return documentos


def normalizar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(char for char in texto if unicodedata.category(char) != "Mn")
    texto = re.sub(r"[^a-z0-9ñ]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def singularizar_basico(palabra):
    if len(palabra) > 5 and palabra.endswith("iones"):
        return palabra[:-5] + "ion"
    if len(palabra) > 4 and palabra.endswith("es"):
        return palabra[:-2]
    if len(palabra) > 3 and palabra.endswith("s"):
        return palabra[:-1]
    return palabra


def extraer_terminos(texto):
    texto = normalizar_texto(texto)
    return {
        singularizar_basico(palabra)
        for palabra in re.findall(r"[a-z0-9ñ]+", texto)
        if len(palabra) > 2 and palabra not in STOPWORDS
    }


def tiene_coincidencia_lexica(pregunta, contenido):
    """Evita enviar a Groq fragmentos recuperados solo por similitud débil."""
    terminos_pregunta = extraer_terminos(pregunta)
    if not terminos_pregunta:
        return False

    terminos_contenido = extraer_terminos(contenido)
    return bool(terminos_pregunta.intersection(terminos_contenido))


def normalizar_palabra_clave(palabra):
    palabra = normalizar_texto(palabra)
    partes = [
        singularizar_basico(parte)
        for parte in palabra.split()
        if parte and parte not in STOPWORDS
    ]
    return " ".join(partes)


def detectar_intenciones(pregunta):
    pregunta_normalizada = normalizar_texto(pregunta)
    terminos_pregunta = extraer_terminos(pregunta)
    intenciones_detectadas = []

    for nombre, configuracion in INTENCIONES.items():
        coincidencias = []

        for palabra in configuracion["palabras"]:
            palabra_normalizada = normalizar_palabra_clave(palabra)
            if not palabra_normalizada:
                continue

            if " " in palabra_normalizada:
                if palabra_normalizada in pregunta_normalizada:
                    coincidencias.append(palabra)
            elif palabra_normalizada in terminos_pregunta:
                coincidencias.append(palabra)

        if coincidencias:
            intenciones_detectadas.append(
                {
                    "intencion": nombre,
                    "palabras": coincidencias,
                    "secciones": configuracion["secciones"],
                }
            )

    if len(intenciones_detectadas) > 1:
        intenciones_detectadas = [
            item
            for item in intenciones_detectadas
            if not (
                item["intencion"] == "requisitos"
                and set(item["palabras"]) == {"necesito"}
            )
        ]

    return intenciones_detectadas


def seleccionar_secciones_por_palabras_clave(pregunta, secciones):
    """Prioriza secciones completas usando intención antes que similitud semántica."""
    documentos = []
    titulos_agregados = set()
    intenciones = detectar_intenciones(pregunta)

    print("[RAG] Intenciones detectadas:", [item["intencion"] for item in intenciones])
    print(
        "[RAG] Palabras clave encontradas:",
        {item["intencion"]: item["palabras"] for item in intenciones},
    )

    secciones_priorizadas = []
    for item in intenciones:
        secciones_priorizadas.extend(item["secciones"])

    print("[RAG] Secciones priorizadas:", secciones_priorizadas)

    for titulo_objetivo in secciones_priorizadas:
        for seccion in secciones:
            titulo = extraer_titulo_seccion(seccion)
            if titulo in titulos_agregados:
                continue

            if normalizar_texto(titulo_objetivo) == normalizar_texto(titulo):
                documentos.append(
                    SimpleNamespace(
                        page_content=seccion,
                        metadata={"titulo": titulo, "origen": "intencion"},
                    )
                )
                titulos_agregados.add(titulo)

    return documentos


def seleccionar_documentos_relevantes(
    resultados,
    pregunta,
    umbral_distancia=1.8,
    secciones=None,
):
    """
    Recibe resultados de Chroma como (documento, distancia).
    Chroma usa distancia: mientras menor sea el valor, más relevante es el fragmento.
    """
    documentos = []
    titulos_agregados = set()

    for documento in seleccionar_secciones_por_palabras_clave(pregunta, secciones or []):
        titulo = getattr(documento, "metadata", {}).get("titulo", "")
        documentos.append(documento)
        titulos_agregados.add(titulo)

    for documento, distancia in resultados:
        titulo = getattr(documento, "metadata", {}).get("titulo", "")
        if titulo in titulos_agregados:
            continue

        if distancia <= umbral_distancia and tiene_coincidencia_lexica(
            pregunta,
            documento.page_content,
        ):
            documentos.append(documento)
            titulos_agregados.add(titulo)

    return documentos


def construir_contexto(documentos):
    return "\n\n---\n\n".join(documento.page_content.strip() for documento in documentos)


def construir_mensajes_groq(contexto, pregunta):
    user_prompt = f"""
CONTEXTO:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

RESPUESTA:
""".strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
