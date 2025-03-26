"""
RAG 시스템 체인 생성 관련 함수
"""

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def create_question_router(llm):
    """
    질문 라우터 체인을 생성합니다.

    Args:
        llm: LLM 모델

    Returns:
        chain: 질문 라우터 체인
    """
    system = """You are an expert at routing a user question to a vectorstore or web search.
Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks.
You do not need to be stringent with the keywords in the question related to these topics.
Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}"),
        ]
    )

    return prompt | llm | JsonOutputParser()


def create_retrieval_grader(llm):
    """
    검색 결과 평가기 체인을 생성합니다.

    Args:
        llm: LLM 모델

    Returns:
        chain: 검색 평가기 체인
    """
    system = """You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n document: {document} "),
        ]
    )

    return prompt | llm | JsonOutputParser()


def create_rag_chain(llm):
    """
    RAG 생성 체인을 생성합니다.

    Args:
        llm: LLM 모델

    Returns:
        chain: RAG 체인
    """
    system = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n context: {context} "),
        ]
    )

    return prompt | llm | StrOutputParser()


def create_hallucination_grader(llm):
    """
    환각 평가기 체인을 생성합니다.

    Args:
        llm: LLM 모델

    Returns:
        chain: 환각 평가기 체인
    """
    system = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "documents: {documents}\n\n answer: {generation} "),
        ]
    )

    return prompt | llm | JsonOutputParser()
