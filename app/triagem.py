"""Triagem de entrada (Model Cascading): guardrail + classificação de intenção.

Camada determinística de baixíssima latência (regras/regex, sem chamada de rede) que
roda ANTES do LangGraph/Gemini:

1. Guardrail de entrada: barra pedidos claramente proibidos (margem de outras
   revendedoras, senhas/credenciais, PII de terceiros, tentativas explícitas de
   jailbreak) sem gastar tempo nem cota de LLM.
2. Classificação de intenção: RAG | PEDIDO | HANDOFF (ou GERAL quando nenhuma regra
   casa), usada para guiar o grafo.

Limites conhecidos: regras só pegam tentativas *explícitas* e escritas de forma
previsível. Ataques sutis/ofuscados continuam cobertos pelas demais camadas
(system prompt, autorização na camada de dados e guardrail de saída, ver README).
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("uvicorn.error")

Intencao = Literal["RAG", "PEDIDO", "HANDOFF", "GERAL"]

MSG_BLOQUEIO_DADOS = (
    "Desculpe, não posso compartilhar informações sobre margens de lucro ou dados "
    "confidenciais. Posso ajudar com as promoções do ciclo e com o status dos seus pedidos."
)
MSG_BLOQUEIO_GERAL = (
    "Desculpe, não posso atender a esse tipo de solicitação. "
    "Posso ajudar com as promoções do ciclo e com o status dos seus pedidos."
)


def _normalizar(texto: str) -> str:
    """minúsculas, sem acentos e com espaços colapsados (regras ficam mais simples)."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento.lower()).strip()


def _compilar(padroes: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in padroes]


# ---------------------------------------------------------------------------
# 1) Guardrail de entrada
# ---------------------------------------------------------------------------
_OUTRA = r"(outra|outras|outro|outros|vizinha|colega|concorrente|fulana|sicrana|terceir[ao]s?)"
_FINANCEIRO = r"(margem|lucro|lucros|comissao|comissoes|ganho|ganhos|ganha|ganhando|faturamento|fatura|faturou|desempenho)"

# (categoria, mensagem, padrões) — padrões rodam sobre o texto normalizado.
_REGRAS_GUARDRAIL: list[tuple[str, str, list[re.Pattern]]] = [
    (
        "margem_de_terceiros",
        MSG_BLOQUEIO_DADOS,
        _compilar([
            rf"\b{_FINANCEIRO}\b.{{0,60}}\b{_OUTRA}\b",
            rf"\b{_OUTRA}\b.{{0,60}}\b{_FINANCEIRO}\b",
            rf"\b{_FINANCEIRO}\b.{{0,40}}\bde (todas|qualquer) (as )?revendedoras?\b",
            # "a revendedora do meu bairro/da minha equipe/da minha cidade..."
            rf"\b{_FINANCEIRO}\b.{{0,40}}\brevendedoras? (do|da|de|dos|das) (meu|minha|nosso|nossa|outro|outra)? ?(bairro|cidade|regiao|rua|vizinhanca|equipe|grupo|rede|lider)\b",
            r"\branking de (vendas|revendedoras|faturamento)\b",
        ]),
    ),
    (
        "credenciais",
        MSG_BLOQUEIO_DADOS,
        _compilar([
            r"\b(senhas?|credenciais|tokens?|api ?keys?|chaves? de api)\b.{0,30}\b(do sistema|de admin|do admin|administrador|do banco|do servidor|da api|de acesso|dos? (funcionarios|atendentes|usuarios)|de outr[ao]s?)\b",
            r"\b(passe|passa|manda|mande|mostre|mostra|revele|informe|envie|fale|me d[eê]|me da)\b.{0,25}\b(senhas?|credenciais|api ?keys?)\b",
            r"\bacesso (de|como) (admin|administrador|root)\b",
        ]),
    ),
    (
        "pii_de_terceiros",
        MSG_BLOQUEIO_DADOS,
        _compilar([
            rf"\b(cpf|telefone|celular|endereco|e-?mail|dados pessoais|dados bancarios|dados de pagamento|cartao)\b.{{0,40}}\b(d[aoe]s? )?{_OUTRA}\b",
            r"\b(cpf|telefone|celular|endereco|e-?mail|dados pessoais|dados bancarios)\b.{0,30}\bd[aoe]s? (clientes?|consumidores?|revendedoras?|usuarios?)\b",
            r"\b(lista|base|planilha) de (cpfs?|clientes|revendedoras|telefones|enderecos)\b",
        ]),
    ),
    (
        "jailbreak",
        MSG_BLOQUEIO_GERAL,
        _compilar([
            r"\b(ignore|ignora|desconsidere|esqueca|esquece|desative|burle)\b.{0,30}\b(instrucoes|regras|restricoes|diretrizes|guardrails?|seguranca|prompt)\b",
            r"\bignore (all |the )?(previous|prior|above) (instructions|rules)\b",
            r"\b(modo|mode) (desenvolvedor|developer|deus|god|dan)\b",
            r"\bjailbreak\b|\bdo anything now\b",
            r"\b(revele|mostre|mostra|exiba|imprima|repita|me diga|qual (e|eh))\b.{0,25}\b(seu|o seu|o|teu)?\s?(system prompt|prompt de sistema|prompt do sistema|instrucoes internas|instrucoes iniciais)\b",
            r"\b(finja|aja como|voce agora e|a partir de agora voce)\b.{0,50}\b(sem regras|sem restricoes|sem limites|sem filtro|nao tem (regras|restricoes))\b",
        ]),
    ),
]


@dataclass(frozen=True)
class ResultadoGuardrail:
    bloqueado: bool
    categoria: str | None = None
    mensagem: str | None = None


def verificar_guardrail(mensagem: str) -> ResultadoGuardrail:
    """Aplica as regras de segurança à mensagem do usuário (rápido e determinístico)."""
    texto = _normalizar(mensagem)
    for categoria, resposta, padroes in _REGRAS_GUARDRAIL:
        if any(p.search(texto) for p in padroes):
            return ResultadoGuardrail(True, categoria, resposta)
    return ResultadoGuardrail(False)


# ---------------------------------------------------------------------------
# 2) Classificação de intenção
# ---------------------------------------------------------------------------
_HANDOFF = _compilar([
    # pedido explícito por atendimento humano
    r"\b(falar|conversar|atendimento|atendente|ser atendid[ao]) .{0,25}\b(humano|atendente|pessoa|especialista|supervisor|gerente|gente de verdade|alguem de verdade)\b",
    r"\b(quero|preciso|gostaria de|me passa|me passe|me transfere|me transfira|transfere|transfira|chama|chame)\b.{0,25}\b(atendente|humano|supervisor|gerente|especialista|pessoa de verdade)\b",
    r"\batendimento humano\b",
    r"\bhumano\b.{0,15}\b(por favor|pfv|agora)\b",
    # frustração explícita
    r"\bja (pedi|falei|expliquei|liguei|tentei)\b.{0,20}\b(mil|varias|muitas|diversas|inumeras) vezes\b",
    r"\b(isso|isto|tudo) (e|eh) (um )?(absurdo|ridiculo|palhacada|descaso|vergonha|inaceitavel)\b",
    r"\bvoces nao resolvem nada\b|\bninguem resolve\b",
    r"\bpessimo atendimento\b|\batendimento (horrivel|pessimo|lixo)\b",
    r"\b(vou|irei) (reclamar|processar|denunciar|cancelar tudo)\b|\bprocon\b|\breclame aqui\b",
    r"\b(estou|to) (muito |super )?(irritada|irritado|indignada|indignado|furiosa|furioso|revoltada|revoltado)\b",
])

_PEDIDO = _compilar([
    r"\bped-?\d{4}-?\d{3,}\b",
    r"\b(pedido|pedidos|rastreio|rastrear|rastreamento|entrega|entregas|entregue|transportadora|codigo de rastreio|frete)\b",
    r"\b(onde (esta|ta|anda)|cade|quando (chega|vai chegar)|chegou|atrasad[oa]|status)\b",
    r"\bmeu cpf\b|\bcpf\b",
])

_RAG = _compilar([
    r"\b(promocao|promocoes|promo|desconto|descontos|oferta|ofertas|catalogo|ciclo ?(15|16)|ciclo)\b",
    r"\b(preco|precos|valor|quanto (custa|e|fica|sai))\b",
    r"\b(produto|produtos|perfume|perfumes|colonia|shampoo|hidratante|kit|brinde|estoque|lancamento)\b",
    r"\b(malbec|lily|nativa|niina|siage|boticario|eudora)\b",
    r"\b(margem|lucro) (do|da|de|dos|das) (produto|perfume|malbec|lily|nativa|niina|siage)\b",
])


def classificar_intencao(mensagem: str) -> Intencao:
    """Classifica a intenção primária por regras. HANDOFF tem prioridade absoluta;
    entre PEDIDO e RAG vence quem tiver mais indícios (empate favorece PEDIDO, pois
    exige dados do usuário). Sem indícios, devolve GERAL e o agente decide sozinho."""
    texto = _normalizar(mensagem)
    if any(p.search(texto) for p in _HANDOFF):
        return "HANDOFF"
    pontos_pedido = sum(bool(p.search(texto)) for p in _PEDIDO)
    pontos_rag = sum(bool(p.search(texto)) for p in _RAG)
    if pontos_pedido == 0 and pontos_rag == 0:
        return "GERAL"
    return "PEDIDO" if pontos_pedido >= pontos_rag else "RAG"


# ---------------------------------------------------------------------------
# Orquestração da triagem (usada pela rota /chat)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResultadoTriagem:
    bloqueado: bool
    intencao: Intencao | None = None
    resposta: str | None = None  # mensagem segura a enviar ao usuário quando bloqueado
    categoria: str | None = None


async def triar_entrada(mensagem: str, thread_id: str) -> ResultadoTriagem:
    """Guardrail de entrada + classificação de intenção, com logs estruturados.

    Assíncrona para se encaixar no fluxo do FastAPI; o trabalho em si é CPU-leve
    (microssegundos), então não precisa de threads."""
    guardrail = verificar_guardrail(mensagem)
    if guardrail.bloqueado:
        logger.warning(
            "[GUARDRAIL] Bloqueio acionado na entrada para thread_id=%s. "
            "Requisição encerrada sem chamar LLM gerador.",
            thread_id,
        )
        logger.debug("[GUARDRAIL] categoria=%s", guardrail.categoria)
        return ResultadoTriagem(True, None, guardrail.mensagem, guardrail.categoria)

    intencao = classificar_intencao(mensagem)
    logger.info("[TRIAGEM] Intenção detectada: %s | Guardrail: PASSED", intencao)
    return ResultadoTriagem(False, intencao)
