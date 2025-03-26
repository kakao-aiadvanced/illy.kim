"""
RAG 시스템의 모델 및 상태 정의
"""

from typing import List
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    그래프의 상태를 나타냅니다.

    속성:
        question: 질문
        generation: LLM 생성 결과
        web_search: 웹 검색을 추가할지 여부
        documents: 문서 목록
    """

    question: str  # 질문 (문자열)
    generation: str  # LLM이 생성한 답변 (문자열)
    web_search: str  # 웹 검색을 추가할지 여부 (문자열, 'Yes' 또는 'No')
    documents: List[str]  # 문서 목록
