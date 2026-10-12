"""Utilitários de resiliência: classificação de erros transitórios e retomada do grafo."""

import httpx
from google.genai.errors import ServerError

# Política de retentativa da rota /chat: 1 chamada original + 2 retries.
MAX_TENTATIVAS = 3
BACKOFF_INICIAL_S = 0.5  # espera antes do 1º retry; dobra a cada novo retry (0.5s, 1s)

# Erros de rede/timeout/indisponibilidade do servidor que podem funcionar numa nova tentativa.
_ERROS_TRANSITORIOS = (
    ConnectionError,  # inclui ConnectionResetError, BrokenPipeError etc.
    TimeoutError,  # em Python 3.11+ também é asyncio.TimeoutError
    httpx.TransportError,  # ConnectError, ReadTimeout, ConnectTimeout, RemoteProtocolError...
    ServerError,  # respostas 5xx do Gemini (ex.: 503 indisponível, 504 timeout)
)


def eh_transitorio(erro: BaseException) -> bool:
    """True se o erro (ou sua causa) for de conexão/timeout e valer a pena tentar de novo.

    Não são retentados: 429 (cota esgotada: tentar de novo só gasta mais cota),
    4xx em geral (chave inválida, modelo inexistente, requisição inválida) e falhas
    de certificado SSL (permanentes)."""
    visto: set[int] = set()
    atual: BaseException | None = erro
    while atual is not None and id(atual) not in visto:
        visto.add(id(atual))
        if "CERTIFICATE_VERIFY_FAILED" in str(atual):
            return False
        if isinstance(atual, _ERROS_TRANSITORIOS):
            return True
        atual = atual.__cause__ or atual.__context__
    return False


async def entrada_para_retry(grafo, config: dict, entrada_original):
    """Define a entrada da próxima tentativa sem duplicar a mensagem do usuário.

    O LangGraph grava a mensagem do usuário no checkpoint antes de executar os nós.
    Se a falha ocorreu depois disso, há nós pendentes (`state.next`): retomamos de
    onde parou passando `None` (não reenvia a mensagem, e não refaz tools já executadas).
    Se nada foi gravado, reenviamos a entrada original."""
    estado = await grafo.aget_state(config)
    return None if estado.next else entrada_original
