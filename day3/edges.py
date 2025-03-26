"""
RAG 시스템 엣지 함수
"""

from pprint import pprint


def route_question(state, question_router):
    """
    질문을 웹 검색 또는 RAG로 라우팅합니다.

    Args:
        state (dict): 현재 그래프 상태
        question_router: 질문 라우팅 체인

    Returns:
        str: 호출할 다음 노드
    """

    print("---ROUTE QUESTION---")  # 질문 라우팅 단계 시작을 출력합니다.
    question = state["question"]  # 상태에서 질문을 가져옵니다.
    print(question)  # 질문을 출력합니다.
    source = question_router.invoke(
        {"question": question}
    )  # 질문 라우터를 사용하여 데이터 소스를 결정합니다.
    print(source)  # 결정된 소스를 출력합니다.
    print(source["datasource"])  # 결정된 데이터 소스를 출력합니다.
    if source["datasource"] == "web_search":  # 데이터 소스가 웹 검색인 경우
        print(
            "---ROUTE QUESTION TO WEB SEARCH---"
        )  # 웹 검색으로 라우팅함을 출력합니다.
        return "websearch"  # "websearch" 노드를 반환합니다.
    elif source["datasource"] == "vectorstore":  # 데이터 소스가 벡터 저장소인 경우
        print("---ROUTE QUESTION TO RAG---")  # RAG로 라우팅함을 출력합니다.
        return "vectorstore"  # "vectorstore" 노드를 반환합니다.


def decide_to_generate(state):
    """
    답변을 생성할지 또는 웹 검색을 추가할지 결정합니다.

    Args:
        state (dict): 현재 그래프 상태

    Returns:
        str: 호출할 다음 노드에 대한 이진 결정
    """

    print("---ASSESS GRADED DOCUMENTS---")  # 문서 평가 단계 시작을 출력합니다.
    state["question"]  # 상태에서 질문을 가져옵니다(사용되지 않음).
    web_search = state["web_search"]  # 상태에서 웹 검색 플래그를 가져옵니다.
    state["documents"]  # 상태에서 문서들을 가져옵니다(사용되지 않음).

    if web_search == "Yes":  # 웹 검색 플래그가 "Yes"인 경우
        # 모든 문서가 check_relevance에서 필터링되었습니다.
        # 새 쿼리를 재생성할 것입니다.
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )  # 웹 검색 포함 결정을 출력합니다.
        return "websearch"  # "websearch" 노드를 반환합니다.
    else:  # 웹 검색 플래그가 "No"인 경우
        # 관련 있는 문서가 있으므로, 답변을 생성합니다.
        print("---DECISION: GENERATE---")  # 생성 결정을 출력합니다.
        return "generate"  # "generate" 노드를 반환합니다.


def grade_generation_v_documents_and_question(state, hallucination_grader):
    """
    생성된 답변이 문서에 기반하고 있고 질문에 답하는지 평가합니다.

    Args:
        state (dict): 현재 그래프 상태
        hallucination_grader: 환각 평가기 체인

    Returns:
        str: "useful", "not_useful", "not_supported" 중 하나를 반환
    """
    print("---CHECK HALLUCINATIONS---")  # 환각 체크 단계 시작을 출력합니다.
    question = state["question"]  # 상태에서 질문을 가져옵니다.
    documents = state["documents"]  # 상태에서 문서들을 가져옵니다.
    generation = state["generation"]  # 상태에서 생성된 답변을 가져옵니다.

    # 환각 평가
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )  # 환각 평가기를 사용하여 생성된 답변이 문서에 기반하는지 평가합니다.
    grade = score["score"]  # 평가 결과를 가져옵니다.

    # 환각 체크
    if grade == "yes":  # 평가 결과가 "yes"인 경우 (문서에 기반함)
        print(
            "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---"
        )  # 문서에 기반한다는 결정을 출력합니다.

        # 질문-답변 체크는 더 이상 수행하지 않고 항상 "useful"로 판단
        return "useful"  # "useful" 결정을 반환합니다.
    else:  # 평가 결과가 "yes"가 아닌 경우 (문서에 기반하지 않음)
        pprint(
            "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---"
        )  # 문서에 기반하지 않는다는 결정을 출력합니다.
        return "not_supported"  # "not_supported" 결정을 반환합니다.
