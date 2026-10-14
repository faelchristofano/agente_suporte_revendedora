from pathlib import Path

from dotenv import load_dotenv

# Carrega o .env antes de importar qualquer módulo que inicialize modelos/chaves de API.
load_dotenv(Path(__file__).resolve().parent / ".env")

import asyncio  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langfuse import get_client  # noqa: E402

# Langfuse >= 3: o CallbackHandler fica em `langfuse.langchain`
# (o caminho antigo `langfuse.callback` só existe na v2).
from langfuse.langchain import CallbackHandler  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.agent.graph import app as graph_app  # noqa: E402
from app.resilience import (  # noqa: E402
    BACKOFF_INICIAL_S,
    MAX_TENTATIVAS,
    eh_transitorio,
    entrada_para_retry,
)
from app.tools.rag import aquecer_indice  # noqa: E402
from app.triagem import triar_entrada  # noqa: E402


logger = logging.getLogger("uvicorn.error")


def _mensagem_de_erro(e: BaseException) -> str:
    """Mensagem útil para o cliente. `str(e)` pode ser vazio (ex.: timeouts),
    então sempre inclui o nome da exceção."""
    detalhe = str(e).strip()
    if "429" in detalhe or "RESOURCE_EXHAUSTED" in detalhe or "RateLimit" in type(e).__name__:
        return (
            "Limite de uso do modelo Gemini excedido (429). "
            "Aguarde alguns minutos ou troque o modelo/chave. "
        )
    return f"{type(e).__name__}: {detalhe}" if detalhe else type(e).__name__


# Observabilidade (Langfuse): ativa somente se as chaves estiverem no .env.
# O CallbackHandler lê LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY e LANGFUSE_HOST do ambiente.
LANGFUSE_ATIVO = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)
if not LANGFUSE_ATIVO:
    print("[startup] Langfuse desativado: defina LANGFUSE_PUBLIC_KEY e LANGFUSE_SECRET_KEY.")


# Chaves para ligar/desligar as estratégias de estabilidade (útil para testes/diagnóstico).
# DESATIVADOS TEMPORARIAMENTE: para reativar, basta trocar para True.
WARMUP_ATIVO = True  # warm-up no startup (ChromaDB + ping no LangGraph/Gemini)
RETRY_ATIVO = True  # retry automático na rota /chat (3 tentativas)

WARMUP_TIMEOUT_S = 60
WARMUP_THREAD_ID = "__warmup__"  # thread isolada: o ping não aparece nas conversas reais


async def _warmup() -> None:
    """Aquece a aplicação ANTES de receber tráfego, para eliminar o cold start.

    1. Carrega o catálogo no ChromaDB em memória (gera os embeddings).
    2. Faz uma invocação "fantasma" (ping) no LangGraph, forçando o handshake
       TCP/TLS e a autenticação com a API do Gemini (a 1ª chamada real é a mais lenta).

    Falhas aqui NÃO derrubam o servidor (ex.: cota 429 ou rede instável): o erro é
    logado e a API sobe; a primeira requisição real pagará o custo de aquecimento."""
    try:
        t0 = time.perf_counter()
        await asyncio.wait_for(asyncio.to_thread(aquecer_indice), WARMUP_TIMEOUT_S)
        logger.info("[warm-up] ChromaDB carregado em memória (%.1fs)", time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001
        logger.warning("[warm-up] Falha ao carregar o ChromaDB: %s: %s", type(e).__name__, e)

    try:
        t0 = time.perf_counter()
        await asyncio.wait_for(
            graph_app.ainvoke(
                {"messages": [("user", "Responda apenas: pong")]},
                config={"configurable": {"thread_id": WARMUP_THREAD_ID}},
            ),
            WARMUP_TIMEOUT_S,
        )
        logger.info("[warm-up] Ping no LangGraph/Gemini concluído (%.1fs)", time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001
        logger.warning("[warm-up] Falha no ping do LangGraph/Gemini: %s: %s", type(e).__name__, e)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if WARMUP_ATIVO:
        logger.info("Iniciando warm-up...")
        t0 = time.perf_counter()
        await _warmup()  # o servidor só começa a aceitar requisições depois daqui
        logger.info("Servidor pronto (warm-up em %.1fs)", time.perf_counter() - t0)
    else:
        logger.info("Warm-up desativado (WARMUP_ATIVO=False). Servidor pronto.")
    yield
    logger.info("Encerrando servidor...")
    if LANGFUSE_ATIVO:
        # Garante o envio dos traces pendentes antes de encerrar o processo.
        await asyncio.to_thread(get_client().flush)


app = FastAPI(
    title="Agente de Suporte à Revendedora Boticário/Eudora", lifespan=lifespan
)


class ChatRequest(BaseModel):
    thread_id: str
    mensagem: str


def _extrair_texto(content) -> str:
    """O conteúdo do chunk pode ser str ou lista de partes (blocos de texto)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            parte if isinstance(parte, str) else parte.get("text", "")
            for parte in content
            if isinstance(parte, (str, dict))
        )
    return ""


def _formatar_sse(texto: str, evento: str | None = None) -> str:
    """Formata um evento SSE. Cada linha do texto precisa do seu próprio 'data:',
    pois uma quebra de linha solta encerraria o evento prematuramente."""
    linhas = "".join(f"data: {linha}\n" for linha in texto.split("\n"))
    prefixo = f"event: {evento}\n" if evento else ""
    return f"{prefixo}{linhas}\n"


@app.get("/health")
async def health():
    """Health check para orquestradores e load balancers (liveness/readiness).

    Propositalmente leve: não chama LLM, banco vetorial nem serviços externos,
    para responder em milissegundos mesmo sob carga."""
    return {"status": "ok", "message": "API operante"}


@app.post("/chat")
async def chat(request: ChatRequest):
    # Um handler por requisição: cada conversa vira um trace separado no Langfuse.
    langfuse_handler = CallbackHandler() if LANGFUSE_ATIVO else None

    config = {
        "configurable": {"thread_id": request.thread_id},
        "callbacks": [langfuse_handler] if langfuse_handler else [],
        # Agrupa os traces da mesma conversa em uma "Session" no Langfuse.
        "metadata": {"langfuse_session_id": request.thread_id},
    }

    async def gerar_eventos():
        try:
            # Comentário SSE (ignorado pelos clientes): faz o primeiro byte sair
            # imediatamente, enquanto o modelo ainda decide se usa tools.
            yield ": conectado\n\n"

            # Cascata de modelos, etapa 1: triagem determinística (microssegundos, sem LLM).
            triagem = await triar_entrada(request.mensagem, request.thread_id)
            if triagem.bloqueado:
                # Guardrail de entrada: aborta ANTES do LangGraph/Gemini (zero custo de LLM)
                # e responde com uma mensagem segura. A mensagem bloqueada não entra no
                # histórico da conversa.
                yield _formatar_sse(triagem.resposta)
                return

            # Etapa 2: a intenção classificada entra no estado inicial do grafo.
            entrada = {
                "messages": [("user", request.mensagem)],
                "intencao_detectada": triagem.intencao,
            }
            enviou_texto = False

            # Retry Pattern: 1 chamada original + até 2 retries, com backoff exponencial curto.
            tentativas_max = MAX_TENTATIVAS if RETRY_ATIVO else 1
            for tentativa in range(1, tentativas_max + 1):
                try:
                    async for chunk, metadata in graph_app.astream(
                        entrada, config=config, stream_mode="messages"
                    ):
                        # Repassa texto de IA vindo dos nós "agente" (tokens em streaming OU a
                        # mensagem inteira quando a resposta sai do cache do LLM) e "transbordo"
                        # (mensagem fixa do human handoff). O LangGraph não repete a mensagem
                        # completa de quem já foi transmitido em chunks. Ignora ToolMessage etc.
                        no = metadata.get("langgraph_node")
                        if isinstance(chunk, AIMessage) and no in ("agente", "transbordo"):
                            texto = _extrair_texto(chunk.content)
                            if texto:
                                enviou_texto = True
                                yield _formatar_sse(texto)
                    break  # sucesso
                except Exception as e:  # noqa: BLE001
                    # Só retenta erros de conexão/timeout e apenas se nada foi enviado ao
                    # cliente (retentar no meio da resposta duplicaria o texto na tela).
                    if (
                        enviou_texto
                        or not eh_transitorio(e)
                        or tentativa == tentativas_max
                    ):
                        raise
                    espera = BACKOFF_INICIAL_S * 2 ** (tentativa - 1)
                    logger.warning(
                        "Erro transitório no /chat (thread_id=%s, tentativa %d/%d): %s: %s. "
                        "Nova tentativa em %.1fs",
                        request.thread_id, tentativa, tentativas_max,
                        type(e).__name__, e, espera,
                    )
                    await asyncio.sleep(espera)
                    # Retoma do checkpoint (sem duplicar a mensagem do usuário no histórico).
                    entrada = await entrada_para_retry(graph_app, config, entrada)
        except Exception as e:  # noqa: BLE001
            # Traceback completo no log do servidor (antes o erro era engolido).
            logger.exception("Falha ao processar /chat (thread_id=%s)", request.thread_id)
            yield _formatar_sse(
                f"Erro ao processar a mensagem: {_mensagem_de_erro(e)}", evento="error"
            )
        finally:
            yield _formatar_sse("[DONE]", evento="done")

    return StreamingResponse(
        gerar_eventos(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",  # no-transform: impede gzip/buffer em proxies
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", reload=False, reload_excludes=["venv/*"])