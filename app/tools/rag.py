"""Tool de RAG para consultar o catálogo de promoções do Ciclo 15."""

from pathlib import Path

from dotenv import load_dotenv

# Carrega o .env da raiz do projeto ANTES de importar/inicializar qualquer
# componente do LangChain que dependa de chaves de API (ex.: GOOGLE_API_KEY).
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
load_dotenv(RAIZ_PROJETO / ".env")

import threading  # noqa: E402

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: E402
from langchain_text_splitters import (  # noqa: E402
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

CAMINHO_CATALOGO = RAIZ_PROJETO / "catalogo_ciclo_15.md"
MODELO_EMBEDDING = "models/gemini-embedding-001"
TOP_K = 3


def _carregar_chunks(caminho: Path):
    """Lê o markdown e o particiona: primeiro por produto (cabeçalho ##) e,
    se algum produto for grande demais, por tamanho de caracteres."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Catálogo não encontrado em {caminho}. "
            "Execute: python -m app.tools.setup_catalogo"
        )

    texto = caminho.read_text(encoding="utf-8")

    por_secao = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "titulo"), ("##", "produto")],
        strip_headers=False,
    ).split_text(texto)

    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    ).split_documents(por_secao)


_vectorstore: Chroma | None = None
_lock = threading.Lock()


def _get_vectorstore() -> Chroma:
    """Cria (uma única vez por processo) o ChromaDB em memória com o catálogo.

    O lock evita indexar o catálogo em duplicidade se uma pergunta chegar
    enquanto o aquecimento do startup ainda está rodando."""
    global _vectorstore
    with _lock:
        if _vectorstore is None:
            chunks = _carregar_chunks(CAMINHO_CATALOGO)
            _vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING),
                collection_name="catalogo_ciclo_15",
            )  # sem persist_directory => banco 100% em memória
        return _vectorstore


def aquecer_indice() -> None:
    """Constrói o índice vetorial antecipadamente (chamado no startup da API),
    para que a primeira pergunta da revendedora não pague o custo de gerar os
    embeddings do catálogo."""
    _get_vectorstore()


@tool
def consultar_catalogo(query: str) -> str:
    """Consulta o catálogo de promoções do Ciclo 15 de O Boticário e Eudora.

    Use para responder dúvidas da revendedora sobre produtos em promoção:
    preços, descontos, lucro, estoque, descrição, brindes e condições.

    Args:
        query: Pergunta ou termos de busca (ex.: "perfume masculino em promoção").
    """
    try:
        resultados = _get_vectorstore().similarity_search(query, k=TOP_K)
    except FileNotFoundError as e:
        return str(e)

    if not resultados:
        return "Nenhuma promoção encontrada no catálogo do Ciclo 15 para essa busca."

    return "\n\n---\n\n".join(doc.page_content for doc in resultados)


if __name__ == "__main__":
    print(consultar_catalogo.invoke({"query": "Qual o desconto do perfume masculino?"}))
