"""
RAG 시스템 워크플로우 구성 함수
"""

from pprint import pprint
from langgraph.graph import END, StateGraph

from models import GraphState
from nodes import retrieve, generate, grade_documents, web_search
from edges import (
    route_question,
    decide_to_generate,
    grade_generation_v_documents_and_question,
)

__all__ = ["build_workflow", "run_test_query"]


def build_workflow(
    tavily,
    retriever,
    retrieval_grader,
    rag_chain,
    hallucination_grader,
):
    """
    워크플로우 그래프를 구성합니다.

    Args:
        tavily: 웹 검색 클라이언트
        retriever: 문서 검색기
        retrieval_grader: 검색 결과 평가기
        rag_chain: RAG 생성 체인
        hallucination_grader: 환각 평가기

    Returns:
        app: 컴파일된 워크플로우 애플리케이션
    """
    # GraphState 타입의 StateGraph를 생성합니다.
    workflow = StateGraph(GraphState)

    # 각 노드에 대한 래퍼 함수 생성
    def retrieve_with_args(state):
        """검색 노드 래퍼 함수"""
        new_state = retrieve(state, retriever)
        # question을 제외한 새로운 값만 업데이트
        return {"documents": new_state.get("documents", [])}

    def generate_with_args(state):
        """생성 노드 래퍼 함수"""
        new_state = generate(state, rag_chain)
        # question을 제외한 새로운 값만 업데이트
        return {"generation": new_state.get("generation", "")}

    def grade_documents_with_args(state):
        """문서 평가 노드 래퍼 함수"""
        new_state = grade_documents(state, retrieval_grader)
        # question을 제외한 새로운 값만 업데이트
        return {
            "documents": new_state.get("documents", []),
            "web_search": new_state.get("web_search", False),
        }

    def web_search_with_args(state):
        """웹 검색 노드 래퍼 함수"""
        new_state = web_search(state, tavily)
        # question을 제외한 새로운 값만 업데이트
        return {"documents": new_state.get("documents", [])}

    def grade_generation_with_args(state):
        """생성 평가 노드 래퍼 함수"""
        result = grade_generation_v_documents_and_question(state, hallucination_grader)
        # 결과만 반환하고 상태는 변경하지 않음
        return result

    # 노드 정의
    workflow.add_node("websearch", web_search_with_args)  # 웹 검색 노드 추가
    workflow.add_node("retrieve", retrieve_with_args)  # 검색 노드 추가
    workflow.add_node(
        "relevance_checker", grade_documents_with_args
    )  # 문서 평가 노드 추가
    workflow.add_node("generate", generate_with_args)  # 생성 노드 추가

    workflow.set_entry_point("retrieve")

    # "retrieve" 노드에서 "grade_documents" 노드로 가는 엣지 추가
    workflow.add_edge("retrieve", "relevance_checker")

    # "grade_documents" 노드에서 조건부로 다음 노드를 결정하는 엣지 추가
    workflow.add_conditional_edges(
        "relevance_checker",
        decide_to_generate,  # decide_to_generate 함수를 사용하여 다음 노드 결정
        {
            "websearch": "websearch",  # "websearch" 결정이 나오면 "websearch" 노드로 이동
            "generate": "generate",  # "generate" 결정이 나오면 "generate" 노드로 이동
        },
    )

    # "websearch" 노드에서 "relevance_checker" 노드로 가는 엣지 추가
    # 웹 검색 후 문서 평가 시 무한 루프 방지를 위해 상태 확인 로직 추가
    workflow.add_conditional_edges(
        "websearch",
        lambda state: (
            "check_relevance"
            if not state.get("web_search_done", False)
            else "end_not_relevant"
        ),
        {
            "check_relevance": "relevance_checker",  # 첫 번째 웹 검색 후에는 문서 평가로 이동
            "end_not_relevant": END,  # 이미 웹 검색을 시도했으면 종료
        },
    )

    # 웹 검색 상태를 표시하는 노드 추가
    workflow.add_node("mark_web_search_done", lambda state: {"web_search_done": True})

    # 웹 검색 후 상태 표시 노드로 연결
    workflow.add_edge("websearch", "mark_web_search_done")
    workflow.add_edge("mark_web_search_done", "relevance_checker")

    # "generate" 노드에서 조건부로 다음 노드를 결정하는 엣지 추가
    # 환각 감지 시 무한 루프 방지를 위해 상태 확인 로직 추가
    workflow.add_conditional_edges(
        "generate",
        grade_generation_with_args,
        {
            "useful": END,  # "useful" 결정이 나오면 워크플로우 종료
            "not_supported": "mark_hallucination",  # 환각 감지 시 표시 노드로 이동
            "not_useful": "end_hallucination",  # 유용하지 않으면 실패 종료
        },
    )

    # 환각 상태를 표시하는 노드 추가
    workflow.add_node(
        "mark_hallucination", lambda state: {"hallucination_checked": True}
    )

    # 환각 감지 실패 종료 노드 추가
    workflow.add_node(
        "end_hallucination",
        lambda state: {
            "generation": "환각이 감지되었습니다. 답변을 생성할 수 없습니다."
        },
    )

    # 환각 표시 노드에서 환각 실패 종료 노드로 연결
    workflow.add_edge("mark_hallucination", "end_hallucination")

    # 환각 실패 종료 노드에서 워크플로우 종료로 연결
    workflow.add_edge("end_hallucination", END)

    # 워크플로우를 컴파일하여 실행 가능한 애플리케이션으로 만듭니다.
    return workflow.compile()


def run_test_query(app, question="What is prompt?"):
    """
    테스트 질문으로 애플리케이션을 실행합니다.

    Args:
        app: 컴파일된 워크플로우 애플리케이션
        question (str): 테스트 질문

    Returns:
        result: 최종 생성된 답변(문자열)
    """
    # 입력 정의: 질문을 포함하는 딕셔너리
    inputs = {"question": question}

    # 애플리케이션을 스트리밍 모드로 실행하고, 각 출력에 대해 반복합니다.
    last_value = None
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")  # 각 단계가 완료될 때마다 출력합니다.
            last_value = value

    # 최종 생성된 답변을 출력하고 반환합니다.
    if "generation" in last_value:
        print(f"Final answer: {last_value['generation']}")
        return last_value["generation"]
    else:
        print("No generation found in the final state.")
        return "답변을 생성하지 못했습니다."
