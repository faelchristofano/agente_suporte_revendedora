"""Tool de consulta de status de pedidos (banco transacional simulado)."""

from pathlib import Path

from dotenv import load_dotenv

# Carrega o .env da raiz do projeto antes de qualquer inicialização do LangChain.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from typing import Any  # noqa: E402

from langchain_core.tools import tool  # noqa: E402

# Simula uma tabela de pedidos em um banco transacional.
# Chave: (cpf somente dígitos, id_pedido em maiúsculas)
_PEDIDOS_MOCK: dict[tuple[str, str], dict[str, Any]] = {
    ("12345678900", "PED-2026-0001"): {
        "status": "Em rota",
        "transportadora": "Jadlog",
        "codigo_rastreio": "JD123456789BR",
        "data_pedido": "2026-11-03",
        "data_prevista_entrega": "2026-11-10",
        "ultima_atualizacao": "2026-11-08 14:32 - Saiu para entrega ao destinatário",
        "ciclo": 15,
    },
    ("12345678900", "PED-2026-0002"): {
        "status": "Aguardando pagamento",
        "transportadora": None,
        "codigo_rastreio": None,
        "data_pedido": "2026-11-05",
        "data_prevista_entrega": None,
        "ultima_atualizacao": "2026-11-05 09:10 - Boleto gerado, aguardando compensação",
        "ciclo": 15,
    },
    ("98765432100", "PED-2026-0003"): {
        "status": "Entregue",
        "transportadora": "Correios",
        "codigo_rastreio": "BR987654321BR",
        "data_pedido": "2026-10-20",
        "data_prevista_entrega": "2026-10-27",
        "ultima_atualizacao": "2026-10-26 16:05 - Entregue ao destinatário",
        "ciclo": 14,
    },
}


def _somente_digitos(valor: str) -> str:
    return "".join(c for c in valor if c.isdigit())


@tool
def buscar_status_pedido(cpf: str, id_pedido: str) -> dict:
    """Consulta o status e o rastreio de um pedido de uma revendedora.

    QUANDO USAR: use esta ferramenta sempre que a revendedora perguntar sobre um
    pedido já realizado, por exemplo: "onde está meu pedido?", "quando meu pedido
    chega?", "meu pedido já foi enviado?", "qual a transportadora?" ou "qual o
    código de rastreio?".

    QUANDO NÃO USAR: não use para dúvidas sobre produtos, preços, descontos ou
    promoções (para isso use `consultar_catalogo`), nem para criar, alterar ou
    cancelar pedidos.

    PRÉ-REQUISITOS: é obrigatório ter AMBOS os parâmetros. Se a revendedora não
    informou o CPF ou o número do pedido, PERGUNTE a ela antes de chamar a
    ferramenta. Nunca invente ou presuma esses valores.

    Args:
        cpf: CPF da revendedora titular do pedido, como string. Aceita com ou sem
            pontuação ("123.456.789-00" ou "12345678900"); deve ter 11 dígitos.
            Usado para garantir que a revendedora só acesse os próprios pedidos.
        id_pedido: Identificador do pedido, como string, no formato "PED-AAAA-NNNN"
            (ex.: "PED-2026-0001"). Maiúsculas e minúsculas são indiferentes.

    Returns:
        Um dicionário. Em caso de sucesso, contém:
            - sucesso (bool): True.
            - id_pedido (str): identificador do pedido.
            - status (str): situação atual, por exemplo "Aguardando pagamento",
              "Em rota" ou "Entregue".
            - transportadora (str | None): empresa responsável pela entrega.
            - codigo_rastreio (str | None): código para rastreamento.
            - data_pedido (str): data da compra (AAAA-MM-DD).
            - data_prevista_entrega (str | None): previsão de entrega (AAAA-MM-DD).
            - ultima_atualizacao (str): último evento registrado.
            - ciclo (int): ciclo de vendas do pedido.
        Em caso de falha, contém sucesso=False e o campo `erro` com o motivo
        (CPF inválido ou pedido não encontrado). Informe o motivo à revendedora
        e peça para conferir os dados.
    """
    cpf_limpo = _somente_digitos(cpf)
    if len(cpf_limpo) != 11:
        return {
            "sucesso": False,
            "erro": "CPF inválido: informe os 11 dígitos do CPF da revendedora.",
        }

    id_normalizado = id_pedido.strip().upper()
    pedido = _PEDIDOS_MOCK.get((cpf_limpo, id_normalizado))
    if pedido is None:
        return {
            "sucesso": False,
            "erro": (
                f"Pedido {id_normalizado} não encontrado para o CPF informado. "
                "Confira o número do pedido e o CPF."
            ),
        }

    return {"sucesso": True, "id_pedido": id_normalizado, **pedido}


if __name__ == "__main__":
    print(
        buscar_status_pedido.invoke(
            {"cpf": "123.456.789-00", "id_pedido": "PED-2026-0001"}
        )
    )
