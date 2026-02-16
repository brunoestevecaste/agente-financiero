from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.tool_calling_agent.base import create_tool_calling_agent


MODEL_NAME = "gemini-2.5-flash-lite"
SYSTEM_PROMPT = """Eres un analista financiero experto y directo.

OBJETIVO:
1. El usuario te dará el nombre de una empresa/activo o un ticker.
2. Identifica el ticker correcto.
3. Usa SIEMPRE la herramienta `get_asset_weekly_data_and_news` con solo el ticker.
4. Analiza movimiento semanal y relación con noticias del último mes.

RESPUESTA:
- Indica ticker usado.
- Resume evolución del precio semanal.
- Explica 2-4 posibles causas con titulares (fecha + fuente).
- Da veredicto de corto plazo (subida, bajada o lateral) + confianza (alta/media/baja).
- Si hay pocas noticias, reduce confianza.
- Incluye una nota breve de que no es asesoramiento financiero.
"""

POSITIVE_TERMS = [
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
NEGATIVE_TERMS = [
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


st.set_page_config(page_title="Agente Financiero AI", page_icon="📈", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');

    :root {
        --primary: #04332a;
        --bg: #fcfdfc;
        --line: rgba(4, 51, 42, 0.22);
    }

    html, body, [class*="css"] {
        font-family: "Manrope", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 10% 0%, rgba(4, 51, 42, 0.08), transparent 40%),
            radial-gradient(circle at 95% 100%, rgba(4, 51, 42, 0.06), transparent 35%),
            var(--bg);
        color: var(--primary);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .main .block-container {
        max-width: 860px;
        padding-top: 1.7rem;
        padding-bottom: 1.2rem;
    }

    .hero-card {
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(252, 253, 252, 0.9);
        margin-bottom: 0.9rem;
    }

    .hero-title {
        margin: 0;
        color: var(--primary);
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    .hero-subtitle {
        margin: 0.35rem 0 0 0;
        color: var(--primary);
        opacity: 0.8;
        font-size: 0.97rem;
    }

    [data-testid="stSidebar"] {
        background: var(--primary);
    }

    [data-testid="stSidebar"] * {
        color: var(--bg);
    }

    [data-testid="stSidebar"] input {
        background: rgba(252, 253, 252, 0.12) !important;
        color: var(--bg) !important;
        border: 1px solid rgba(252, 253, 252, 0.35) !important;
    }

    [data-testid="stChatInputTextArea"] textarea {
        background: var(--bg) !important;
        color: var(--primary) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatMessageContent"] {
        color: var(--primary);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
      <p class="hero-title">Financial Insight Agent</p>
      <p class="hero-subtitle">Analiza precio semanal y noticias del último mes con enfoque claro y minimalista.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Google API Key", type="password")
    st.caption("La clave solo se usa en esta sesión.")
    if not api_key:
        st.warning("Introduce tu Google API Key para continuar.")
        st.stop()


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


@st.cache_data(ttl=600)
def _get_asset_weekly_data_and_news_text(ticker: str) -> str:
    ticker = ticker.strip().upper()
    stock = yf.Ticker(ticker)

    hist = stock.history(period="7d", interval="1d", auto_adjust=False)
    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        return f"No hay datos recientes para el ticker '{ticker}'."

    hist = hist.tail(7)

    currency = "USD"
    try:
        currency = (stock.fast_info or {}).get("currency") or currency
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

    score = 0
    for article in recent_news:
        text = f"{article['title']} {article['summary']}".lower()
        score += sum(1 for term in POSITIVE_TERMS if term in text)
        score -= sum(1 for term in NEGATIVE_TERMS if term in text)

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
        news_block = "- No se encontraron noticias recientes en Yahoo Finance."

    week_change_text = f"{week_change:+.2f}%" if week_change is not None else "N/D"
    first_close_text = f"{first_close:.2f} {currency}" if first_close is not None else "N/D"
    last_close_text = f"{last_close:.2f} {currency}" if last_close is not None else "N/D"
    week_high_text = f"{week_high:.2f} {currency}" if week_high is not None else "N/D"
    week_low_text = f"{week_low:.2f} {currency}" if week_low is not None else "N/D"

    return (
        f"Ticker analizado: {ticker}\n"
        f"Resumen semanal:\n"
        f"- Cierre inicial: {first_close_text}\n"
        f"- Cierre final: {last_close_text}\n"
        f"- Variación semanal: {week_change_text}\n"
        f"- Máximo semanal: {week_high_text}\n"
        f"- Mínimo semanal: {week_low_text}\n\n"
        f"Serie diaria:\n{chr(10).join(daily_lines)}\n\n"
        f"Noticias del último mes (máx 8):\n{news_block}\n\n"
        f"Indicador heurístico de sesgo por noticias: {news_bias} (score={score})."
    )


@tool
def get_asset_weekly_data_and_news(ticker: str) -> str:
    """Devuelve precios diarios de la última semana y noticias del último mes para un ticker."""
    try:
        return _get_asset_weekly_data_and_news_text(ticker)
    except Exception as e:
        return f"Error al obtener datos para '{ticker}': {e}"


@st.cache_resource
def setup_agent(google_api_key: str) -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
        google_api_key=google_api_key,
    )

    tools = [get_asset_weekly_data_and_news]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Escribe empresa o ticker (ej: Nvidia, BBVA, TSLA, SAN.MC)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                agent_executor = setup_agent(api_key)
                response = agent_executor.invoke(
                    {"input": user_input, "chat_history": st.session_state.chat_history}
                )
                output_text = response["output"]
                st.markdown(output_text)

                st.session_state.messages.append({"role": "assistant", "content": output_text})
                st.session_state.chat_history.extend(
                    [HumanMessage(content=user_input), AIMessage(content=output_text)]
                )
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
