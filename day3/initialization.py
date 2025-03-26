"""
RAG 시스템 초기화 관련 함수
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tavily import TavilyClient

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 환경 변수 로드
load_dotenv()

# USER_AGENT 환경 변수 설정
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "RAG-Demo-App/1.0")


def init_tavily_client(api_key=None):
    """
    Tavily 웹 검색 API 클라이언트를 초기화합니다.

    Args:
        api_key (str, optional): Tavily API 키. 제공되지 않으면 환경 변수에서 로드합니다.

    Returns:
        TavilyClient: 초기화된 Tavily 클라이언트
    """
    # API 키가 제공되지 않은 경우 환경 변수에서 로드
    if api_key is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "Tavily API 키가 환경 변수에 설정되어 있지 않습니다. .env 파일을 확인해주세요."
            )

    return TavilyClient(api_key=api_key)


def init_llm(model_name="gpt-4o-mini", temperature=0):
    """
    LLM 모델을 초기화합니다.

    Args:
        model_name (str): 사용할 OpenAI 모델 이름
        temperature (float): 생성 다양성 조절 (0: 결정적, 1: 다양함)

    Returns:
        ChatOpenAI: 초기화된 LLM 모델
    """
    return ChatOpenAI(model=model_name, temperature=temperature)


def setup_vectorstore(urls=None):
    """
    벡터 스토어를 설정하고 검색기를 반환합니다.

    Args:
        urls (list): 문서를 로드할 URL 목록

    Returns:
        retriever: 벡터 스토어 검색기
    """
    if urls is None:
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

    # 각 URL에서 웹 페이지 로드
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # 벡터 저장소 생성
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="./chroma_db",  # 영구 저장소 디렉토리 지정
    )

    # 변경사항 저장 - Chroma 0.4.x에서는 자동 저장되므로 제거
    # vectorstore.persist()

    return vectorstore.as_retriever()
