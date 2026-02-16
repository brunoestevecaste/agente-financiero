import streamlit as st
import yfinance as yf
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# --- CONFIGURACIÓN DE LA PÁGINA Y ESTILOS ---
st.set_page_config(page_title="Agente Financiero AI", page_icon="📈", layout="centered")

# Paleta de colores: 
# Fondo: #fcfdfc (Blanco roto/claro)
# Acento/Texto principal: #04332a (Verde oscuro profundo)

# Inyectar CSS personalizado
st.markdown("""
    <style>
    /* Fondo general de la aplicación */
    .stApp {
        background-color: #fcfdfc;
        color: #04332a;
    }
    
    /* Input del chat */
    .stChatInput input {
        background-color: #ffffff !important;
        color: #04332a !important;
        border: 1px solid #04332a !important;
    }
    
    /* Títulos y encabezados */
    h1, h2, h3 {
        color: #04332a !important;
        font-family: 'Helvetica', sans-serif;
        font-weight: 300;
    }
    
    /* Mensajes del usuario (Alineados a la derecha, color acento) */
    .st-emotion-cache-janbn0 {
        background-color: #04332a;
        color: #fcfdfc;
    }
    
    /* Mensajes del asistente (Alineados a la izquierda, gris muy suave) */
    .st-emotion-cache-1c7y2kd {
        background-color: #f0f2f0;
        color: #04332a;
    }
    
    /* Botones */
    div.stButton > button {
        background-color: #04332a;
        color: #fcfdfc;
        border-radius: 5px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #064e40;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO ---
st.title("Financial Insight Agent")
st.markdown("Introduce un activo o empresa para revisar precio semanal y noticias del último mes.")

# --- SIDEBAR PARA API KEY ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Google API Key", type="password")
    st.info("Obtén tu clave en [Google AI Studio](https://aistudio.google.com/)")
    if not api_key:
        st.warning("Por favor, introduce tu API Key para continuar.")
        st.stop()

# --- LÓGICA DEL AGENTE ---

def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_news_item(raw_item: Dict[str, Any]) -> Tuple[Optional[datetime], str, str, str, str]:
    title = raw_item.get("title", "")
    summary = raw_item.get("summary", "")
    source = raw_item.get("publisher", "")
    link = raw_item.get("link", "")
    published_at = None

    content = raw_item.get("content")
    if isinstance(content, dict):
        title = content.get("title") or title
        summary = content.get("summary") or summary

        provider = content.get("provider")
        if isinstance(provider, dict):
            source = provider.get("displayName") or source

        canonical_url = content.get("canonicalUrl")
        if isinstance(canonical_url, dict):
            link = canonical_url.get("url") or link

        published_at = _parse_iso_datetime(content.get("pubDate", ""))

    if published_at is None:
        ts = raw_item.get("providerPublishTime")
        if isinstance(ts, (int, float)):
            published_at = datetime.fromtimestamp(ts, tz=timezone.utc)

    if published_at is None:
        published_at = _parse_iso_datetime(raw_item.get("published", ""))

    title = title.strip() if isinstance(title, str) else "Sin titular"
    summary = summary.strip() if isinstance(summary, str) else ""
    source = source.strip() if isinstance(source, str) and source else "Fuente no disponible"
    link = link.strip() if isinstance(link, str) else ""
    return published_at, title or "Sin titular", summary, source, link


@tool
def get_asset_weekly_data_and_news(ticker: str):
    """
    Recibe un TICKER (símbolo bursátil, ej: AAPL, TEF.MC).
    Devuelve precios diarios de la última semana de mercado y noticias del último mes.
    """
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        hist = stock.history(period="7d", interval="1d", auto_adjust=False)
        hist = hist.dropna(subset=["Close"])

        if hist.empty:
            return f"No hay datos de precios recientes para el ticker '{ticker}'."

        hist = hist.tail(7)

        currency = "USD"
        try:
            fast_info = stock.fast_info or {}
            currency = fast_info.get("currency") or currency
        except Exception:
            pass

        first_close = _safe_float(hist.iloc[0].get("Close"))
        last_close = _safe_float(hist.iloc[-1].get("Close"))
        week_change = None
        if first_close not in (None, 0) and last_close is not None:
            week_change = ((last_close - first_close) / first_close) * 100

        week_high = _safe_float(hist["High"].max()) if "High" in hist else None
        week_low = _safe_float(hist["Low"].min()) if "Low" in hist else None

        daily_lines: List[str] = []
        previous_close: Optional[float] = None
        for idx, row in hist.iterrows():
            close_price = _safe_float(row.get("Close"))
            if close_price is None:
                continue

            date_str = idx.strftime("%Y-%m-%d")
            if previous_close in (None, 0):
                daily_lines.append(f"- {date_str}: cierre {close_price:.2f} {currency}")
            else:
                day_pct = ((close_price - previous_close) / previous_close) * 100
                daily_lines.append(
                    f"- {date_str}: cierre {close_price:.2f} {currency} ({day_pct:+.2f}% vs día anterior)"
                )
            previous_close = close_price

        raw_news: List[Dict[str, Any]] = []
        try:
            raw_news = stock.news or []
        except Exception:
            raw_news = []

        if not raw_news and hasattr(stock, "get_news"):
            try:
                raw_news = stock.get_news() or []
            except Exception:
                raw_news = []

        month_ago = datetime.now(tz=timezone.utc) - timedelta(days=30)
        parsed_news = []
        for item in raw_news:
            if not isinstance(item, dict):
                continue
            published_at, title, summary, source, link = _extract_news_item(item)
            if published_at is not None and published_at < month_ago:
                continue
            parsed_news.append(
                {
                    "published_at": published_at,
                    "title": title,
                    "summary": summary,
                    "source": source,
                    "link": link,
                }
            )

        parsed_news.sort(
            key=lambda x: x["published_at"] or datetime(1970, 1, 1, tzinfo=timezone.utc),
            reverse=True,
        )
        recent_news = parsed_news[:8]

        positive_terms = [
            "beat",
            "growth",
            "upgrade",
            "record",
            "strong",
            "profit",
            "rally",
            "buyback",
            "partnership",
            "expansion",
        ]
        negative_terms = [
            "miss",
            "downgrade",
            "lawsuit",
            "investigation",
            "warning",
            "weak",
            "decline",
            "loss",
            "cut",
            "debt",
        ]

        score = 0
        for article in recent_news:
            text = f"{article['title']} {article['summary']}".lower()
            score += sum(1 for term in positive_terms if term in text)
            score -= sum(1 for term in negative_terms if term in text)

        if score >= 2:
            news_bias = "alcista"
        elif score <= -2:
            news_bias = "bajista"
        else:
            news_bias = "mixto/neutral"

        if recent_news:
            news_lines = []
            for article in recent_news:
                date_str = (
                    article["published_at"].astimezone(timezone.utc).strftime("%Y-%m-%d")
                    if article["published_at"]
                    else "fecha no disponible"
                )
                line = f"- {date_str} | {article['source']}: {article['title']}"
                if article["link"]:
                    line += f" ({article['link']})"
                news_lines.append(line)
            news_block = "\n".join(news_lines)
        else:
            news_block = "- No se han encontrado noticias recientes en Yahoo Finance para este activo."

        week_change_text = f"{week_change:+.2f}%" if week_change is not None else "N/D"
        first_close_text = f"{first_close:.2f} {currency}" if first_close is not None else "N/D"
        last_close_text = f"{last_close:.2f} {currency}" if last_close is not None else "N/D"
        week_high_text = f"{week_high:.2f} {currency}" if week_high is not None else "N/D"
        week_low_text = f"{week_low:.2f} {currency}" if week_low is not None else "N/D"

        return (
            f"Ticker analizado: {ticker}\n"
            f"Resumen semanal:\n"
            f"- Cierre inicial (últimos 7 días de mercado): {first_close_text}\n"
            f"- Cierre final (más reciente): {last_close_text}\n"
            f"- Variación semanal: {week_change_text}\n"
            f"- Máximo semanal: {week_high_text}\n"
            f"- Mínimo semanal: {week_low_text}\n"
            f"\n"
            f"Serie diaria (última semana):\n"
            f"{chr(10).join(daily_lines)}\n"
            f"\n"
            f"Noticias del último mes (máx 8):\n"
            f"{news_block}\n"
            f"\n"
            f"Indicador heurístico de sesgo por noticias: {news_bias} (score={score})."
        )
    except Exception as e:
        return f"Error al obtener datos para el ticker '{ticker}'. Asegúrate de que el símbolo sea correcto. Error: {e}"

# Cacheamos el agente para no recrearlo en cada interacción
@st.cache_resource
def setup_agent(google_api_key):
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0
    )
    
    tools = [get_asset_weekly_data_and_news]
    
    # Prompt ingeniería avanzada: Instruimos al modelo para buscar el ticker
    system_prompt = """Eres un analista financiero experto y directo.
    
    TU OBJETIVO:
    1. El usuario te dará el NOMBRE de una empresa/activo o un ticker.
    2. Tu primera tarea mental es identificar el TICKER correcto. 
       - Si es una empresa española, suele terminar en .MC (ej: ITX.MC, SAN.MC).
       - Si es americana, son letras simples (ej: TSLA, AAPL).
    3. USA SIEMPRE la herramienta `get_asset_weekly_data_and_news` pasando SOLO el ticker.
    4. Analiza el movimiento de precios semanal y compáralo con noticias del último mes.
    5. Si no hay noticias suficientes, dilo explícitamente y reduce la confianza.
    
    TU RESPUESTA:
    - Debe ser clara y directa.
    - Indica el ticker usado.
    - Resume cómo ha evolucionado el precio en la última semana.
    - Explica 2-4 posibles causas apoyadas en titulares concretos (fecha + fuente).
    - Da un VEREDICTO final para corto plazo (sesgo: subida, bajada o lateral) y nivel de confianza (alto/medio/bajo).
    - Añade una nota breve de que no es asesoramiento financiero.
    - Usa emojis financieros (📈, 📉, 💰).
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- GESTIÓN DEL CHAT EN STREAMLIT ---

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# Inicializar historial para LangChain (memoria)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("Escribe empresa o ticker (ej: Nvidia, BBVA, TSLA, SAN.MC)..."):
    
    # 1. Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Obtener respuesta del agente
    if api_key:
        agent_executor = setup_agent(api_key)
        
        with st.chat_message("assistant"):
            with st.spinner("Analizando mercados..."):
                try:
                    # Ejecutar agente
                    response = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    output_text = response["output"]
                    st.markdown(output_text)
                    
                    # 3. Guardar en historial
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    
                    # Actualizar memoria de LangChain
                    st.session_state.chat_history.extend([
                        HumanMessage(content=prompt),
                        AIMessage(content=output_text),
                    ])
                    
                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
