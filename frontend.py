"""Interface de chat (Streamlit) do suporte à Revendedora Boticário/Eudora.

Consome o endpoint SSE `POST /chat` do backend FastAPI e renderiza a resposta
em tempo real.

Uso:
    ./venv/bin/python main.py          # terminal 1 (backend, porta 8000)
    ./venv/bin/streamlit run frontend.py   # terminal 2 (interface)
"""

import os
from collections.abc import Iterator

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")
THREAD_ID_PADRAO = "rev-1"
MENSAGEM_BOAS_VINDAS = (
    "Olá! 💚 Sou a assistente virtual da Revendedora Boticário/Eudora. "
    "Posso ajudar com as **promoções do Ciclo 15** e com o **status dos seus pedidos**. "
    "Como posso ajudar?"
)

st.set_page_config(page_title="Suporte à Revendedora", page_icon="💚", layout="centered")


# ---------------------------------------------------------------------------
# Streaming SSE
# ---------------------------------------------------------------------------
def stream_resposta(thread_id: str, mensagem: str) -> Iterator[str]:
    """Faz o POST e entrega o texto da resposta aos poucos (gerador para st.write_stream).

    Formato SSE devolvido pelo backend:
        : conectado              -> comentário (ignorado)
        data: <texto>            -> pedaço da resposta (uma ou mais linhas `data:`)
        event: error / done      -> eventos de controle, terminados por linha em branco
    """
    try:
        resposta = requests.post(
            BACKEND_URL,
            json={"thread_id": thread_id, "mensagem": mensagem},
            stream=True,
            timeout=(5, 120),  # (conexão, espera entre bytes)
        )
        resposta.raise_for_status()
    except requests.ConnectionError:
        yield f"⚠️ Não consegui falar com o servidor em `{BACKEND_URL}`. O backend está rodando?"
        return
    except requests.RequestException as e:
        yield f"⚠️ Erro ao chamar o servidor: {e}"
        return

    evento = "message"
    dados: list[str] = []

    # chunk_size=1: entrega cada linha assim que chega (o padrão espera encher o buffer).
    for bruto in resposta.iter_lines(chunk_size=1):
        linha = bruto.decode("utf-8")  # o SSE não declara charset; força UTF-8

        if linha == "":  # linha em branco = fim do evento
            texto = "\n".join(dados)
            if evento == "done":
                return
            if evento == "error":
                yield f"\n\n⚠️ {texto}"
            elif dados:
                yield texto
            evento, dados = "message", []
        elif linha.startswith(":"):  # comentário/heartbeat
            continue
        elif linha.startswith("event:"):
            evento = linha[len("event:") :].strip()
        elif linha.startswith("data:"):
            valor = linha[len("data:") :]
            # Pela especificação SSE, só UM espaço após ":" é separador; o resto é conteúdo.
            dados.append(valor[1:] if valor.startswith(" ") else valor)


# ---------------------------------------------------------------------------
# Estado: um histórico visual por thread_id
# ---------------------------------------------------------------------------
if "historicos" not in st.session_state:
    st.session_state.historicos = {}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("💚 Suporte Revendedora")
    thread_id = st.text_input(
        "Thread ID (sessão)",
        value=THREAD_ID_PADRAO,
        help="Cada thread_id tem sua própria memória no agente. "
        "Troque o valor para testar o isolamento entre sessões.",
    ).strip() or THREAD_ID_PADRAO
    st.caption(f"Sessão ativa: `{thread_id}`")

    if st.button("🗑️ Limpar tela desta sessão"):
        st.session_state.historicos[thread_id] = []
        st.rerun()

    st.divider()
    st.caption(
        "A memória do agente fica no servidor. "
        "Limpar a tela não apaga o contexto; use outro thread_id para começar do zero."
    )

mensagens: list[dict] = st.session_state.historicos.setdefault(thread_id, [])

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
st.title("Assistente da Revendedora")
st.caption("Boticário · Eudora — Promoções do Ciclo 15 e status de pedidos")

with st.chat_message("assistant", avatar="💚"):
    st.markdown(MENSAGEM_BOAS_VINDAS)

for msg in mensagens:
    with st.chat_message(msg["role"], avatar="💚" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua mensagem..."):
    mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="💚"):
        with st.spinner("Consultando..."):
            fluxo = stream_resposta(thread_id, prompt)
            primeiro = next(fluxo, "")  # o spinner some quando chega o 1º trecho

        def _fluxo_completo() -> Iterator[str]:
            yield primeiro
            yield from fluxo

        resposta_final = st.write_stream(_fluxo_completo())

    mensagens.append({"role": "assistant", "content": resposta_final})
