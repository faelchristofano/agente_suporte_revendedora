"""Tool de transbordo para atendimento humano (Human Handoff)."""

import logging

from langchain_core.tools import tool

logger = logging.getLogger("uvicorn.error")

# Resposta padrão entregue à revendedora quando o transbordo é acionado.
# É fixa (não gerada pelo LLM) para garantir um comportamento previsível.
MENSAGEM_TRANSBORDO = (
    "Compreendo. Estou transferindo seu atendimento para um de nossos especialistas "
    "humanos. Por favor, aguarde um momento."
)

NOME_TOOL_TRANSBORDO = "transferir_para_humano"


@tool(NOME_TOOL_TRANSBORDO)
def transferir_para_humano(motivo: str) -> str:
    """Transfere o atendimento para um atendente humano (transbordo / human handoff).

    QUANDO USAR (acione imediatamente, sem tentar resolver antes):
    - A revendedora pede EXPLICITAMENTE para falar com um humano, atendente,
      pessoa, especialista, supervisor ou "alguém de verdade".
    - A revendedora demonstra EXTREMA frustração, irritação ou insatisfação
      (ex.: xingamentos, ameaças de reclamação formal, "já pedi mil vezes",
      "isso é um absurdo", "vocês não resolvem nada").

    QUANDO NÃO USAR: dúvidas comuns sobre promoções ou pedidos, que as demais
    ferramentas resolvem, nem simples pedidos de esclarecimento ou leve irritação.

    IMPORTANTE: após acionar esta ferramenta, o sistema encerra o atendimento
    automatizado e envia a mensagem padrão de transferência. NÃO escreva mais nada.

    Args:
        motivo: Resumo curto (1 frase) do porquê da transferência, para o atendente
            humano entender o contexto. Ex.: "Revendedora pediu para falar com
            um atendente sobre pedido atrasado". Não inclua CPF nem outros dados
            pessoais.
    """
    # Ponto de integração com a fila de atendimento (Zendesk, Genesys etc.).
    logger.warning("[HANDOFF] Transbordo para humano acionado. Motivo: %s", motivo)
    return "Transbordo registrado. Encerrando o atendimento automatizado."
