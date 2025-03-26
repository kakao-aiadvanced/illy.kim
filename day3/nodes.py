"""
RAG 시스템 노드 함수
"""

from langchain_core.documents import Document


def retrieve(state, retriever):
    """
    벡터 저장소에서 문서를 검색합니다.

    Args:
        state (dict): 현재 그래프 상태
        retriever: 문서 검색기

    Returns:
        state (dict): 문서가 포함된 키가 추가된 상태
    """
    print("---RETRIEVE---")  # 검색 단계 시작을 출력합니다.
    question = state["question"]  # 상태에서 질문을 가져옵니다.

    # 검색 실행
    documents = retriever.invoke(
        question
    )  # 검색기를 사용하여 질문에 관련된 문서를 검색합니다.
    print(question)  # 질문을 출력합니다.
    print(documents)  # 검색된 문서들을 출력합니다.
    return {
        "documents": documents,
        "question": question,
    }  # 검색된 문서와 질문을 포함한 새 상태를 반환합니다.


def generate(state, rag_chain):
    """
    검색된 문서를 기반으로 RAG를 사용하여 답변을 생성합니다.

    Args:
        state (dict): 현재 그래프 상태
        rag_chain: RAG 생성 체인

    Returns:
        state (dict): 생성된 답변이 포함된 키가 추가된 상태
    """
    print("---GENERATE---")  # 생성 단계 시작을 출력합니다.
    question = state["question"]  # 상태에서 질문을 가져옵니다.
    documents = state["documents"]  # 상태에서 문서들을 가져옵니다.

    # RAG 생성
    generation = rag_chain.invoke(
        {"context": documents, "question": question}
    )  # RAG 체인을 사용하여 답변을 생성합니다.
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
    }  # 생성된 답변과 함께 새 상태를 반환합니다.


def grade_documents(state, retrieval_grader):
    """
    검색된 문서가 질문과 관련이 있는지 결정합니다.
    만약 어떤 문서라도 관련이 없다면, 웹 검색을 실행하는 플래그를 설정합니다.

    Args:
        state (dict): 현재 그래프 상태
        retrieval_grader: 문서 관련성 평가기

    Returns:
        state (dict): 관련 없는 문서가 필터링되고 web_search 상태가 업데이트된 상태
    """

    print(
        "---CHECK DOCUMENT RELEVANCE TO QUESTION---"
    )  # 문서 관련성 체크 단계 시작을 출력합니다.
    question = state["question"]  # 상태에서 질문을 가져옵니다.
    documents = state["documents"]  # 상태에서 문서들을 가져옵니다.

    # 각 문서 점수 계산
    filtered_docs = []  # 필터링된 문서를 저장할 빈 리스트를 생성합니다.
    web_search = "No"  # 기본적으로 웹 검색을 하지 않음으로 설정합니다.
    for d in documents:  # 각 문서에 대해 반복합니다.
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )  # 검색 평가기를 사용하여 문서의 관련성을 평가합니다.
        grade = score["score"]  # 평가 결과를 가져옵니다.
        # 문서가 관련 있음
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")  # 문서가 관련 있음을 출력합니다.
            filtered_docs.append(d)  # 필터링된 문서 리스트에 추가합니다.
        # 문서가 관련 없음
        else:
            print(
                "---GRADE: DOCUMENT NOT RELEVANT---"
            )  # 문서가 관련 없음을 출력합니다.
            # 필터링된 문서 리스트에 추가하지 않습니다.
            # 웹 검색을 실행하도록 플래그를 설정합니다.
            web_search = "Yes"  # 웹 검색을 실행하도록 설정합니다.
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }  # 필터링된 문서와 웹 검색 플래그를 포함한 새 상태를 반환합니다.


def web_search(state, tavily):
    """
    질문에 기반한 웹 검색을 수행합니다.

    Args:
        state (dict): 현재 그래프 상태
        tavily: 웹 검색 클라이언트

    Returns:
        state (dict): 웹 결과가 추가된 문서가 포함된 상태
    """

    print("---WEB SEARCH---")  # 웹 검색 단계 시작을 출력합니다.
    print(state)  # 현재 상태를 출력합니다.
    question = state["question"]  # 상태에서 질문을 가져옵니다.
    documents = None  # 문서를 초기에 None으로 설정합니다.
    if "documents" in state:  # 상태에 문서가 있는지 확인합니다.
        documents = state["documents"]  # 상태에서 문서들을 가져옵니다.

    # 웹 검색 수행
    docs = tavily.search(query=question)[
        "results"
    ]  # Tavily API를 사용하여 웹 검색을 수행합니다.
    #    [샘플 검색 결과 주석 - 실제 출력 예시]

    web_results = "\n".join(
        [d["content"] for d in docs]
    )  # 각 검색 결과의 내용을 줄바꿈으로 연결합니다.
    web_results = Document(
        page_content=web_results
    )  # 웹 검색 결과로 Document 객체를 생성합니다.
    if documents is not None:  # 기존 문서가 있는지 확인합니다.
        documents.append(web_results)  # 기존 문서에 웹 검색 결과를 추가합니다.
    else:
        documents = [
            web_results
        ]  # 웹 검색 결과만 포함하는 새 문서 리스트를 생성합니다.
    return {
        "documents": documents,
        "question": question,
    }  # 업데이트된 문서와 질문을 포함한 새 상태를 반환합니다.
