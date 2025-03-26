"""
RAG(Retrieval Augmented Generation) 시스템 메인 실행 파일

이 프로그램은 다음과 같은 주요 기능을 수행합니다:
1. 질문 라우팅: 사용자 질문을 벡터 저장소 또는 웹 검색으로 라우팅
2. 문서 검색: 관련 문서 검색 및 평가
3. 답변 생성: 검색된 문서를 기반으로 답변 생성
4. 품질 평가: 생성된 답변의 품질 평가

전체 흐름:
- 사용자 질문 → 라우팅 결정 → 문서 검색/웹 검색 → 문서 평가 → 답변 생성 → 품질 평가 → 최종 출력
"""

# 필요한 초기화 함수들을 가져옵니다.
# - init_tavily_client: Tavily 웹 검색 API 클라이언트 초기화 (.env 파일에서 API 키 로드)
# - init_llm: LLM 모델 초기화 (기본값: gpt-4o-mini 모델, temperature=0)
# - setup_vectorstore: 벡터 스토어 설정 및 검색기 반환 (문서 로드, 분할, 임베딩)
from initialization import init_tavily_client, init_llm, setup_vectorstore

# 필요한 체인 생성 함수들을 가져옵니다.
# - create_question_router: 질문을 벡터 스토어 또는 웹 검색으로 라우팅하는 체인
# - create_retrieval_grader: 검색된 문서의 관련성 평가 체인 (yes/no 결정)
# - create_rag_chain: 검색된 문서를 바탕으로 답변을 생성하는 체인
# - create_hallucination_grader: 생성된 답변이 문서에 근거하는지 평가하는 체인 (yes/no 결정)
# - create_answer_grader: 생성된 답변이 질문에 유용한지 평가하는 체인 (yes/no 결정)
from chains import (
    create_question_router,
    create_retrieval_grader,
    create_rag_chain,
    create_hallucination_grader,
)

# 워크플로우 관련 함수들을 가져옵니다.
# - build_workflow: 워크플로우 그래프 구성 (노드와 엣지 설정)
# - run_test_query: 테스트 질문으로 워크플로우 실행 (스트리밍 모드)
from workflow import build_workflow, run_test_query


def main():
    """
    메인 애플리케이션 실행 함수

    이 함수는 다음과 같은 순서로 실행됩니다:
    1. 필요한 컴포넌트 초기화 (Tavily, LLM, 벡터 스토어)
    2. 필요한 체인 생성 (라우터, 평가기, 생성기)
    3. 워크플로우 구성
    4. 테스트 질문 실행
    """
    # 1. 초기화
    # ----------------------------------------------------------------
    # Tavily 웹 검색 API 클라이언트를 초기화합니다.
    # .env 파일에서 TAVILY_API_KEY 환경 변수를 로드합니다.
    tavily = init_tavily_client()

    # LLM 모델을 초기화합니다.
    # 기본값: gpt-4o-mini 모델, temperature=0 (결정적 생성)
    llm = init_llm()

    # 벡터 스토어를 설정하고 검색기를 가져옵니다.
    # 1) 지정된 URL에서 문서를 로드하고
    # 2) 문서를 청크로 분할하고
    # 3) 임베딩을 생성하여 벡터 저장소를 만듭니다.
    retriever = setup_vectorstore()

    # # 2. 체인 생성
    # # ----------------------------------------------------------------
    # # 질문 라우팅 체인: 질문을 벡터 스토어 또는 웹 검색으로 라우팅합니다.
    # # - 입력: 질문
    # # - 출력: {"datasource": "vectorstore"} 또는 {"datasource": "web_search"} JSON
    # # - 질문이 벡터 스토어의 주제(에이전트, 프롬프트 엔지니어링, 적대적 공격)와 관련 있으면 vectorstore 선택
    # # - 그렇지 않으면 web_search 선택
    # question_router = create_question_router(llm)
 
    # 검색 결과 평가 체인: 검색된 문서가 질문과 관련 있는지 평가합니다.
    # - 입력: 질문 및 문서
    # - 출력: {"score": "yes"} 또는 {"score": "no"} JSON
    # - 문서가 질문과 관련된 키워드를 포함하면 yes 반환
    retrieval_grader = create_retrieval_grader(llm)

    # RAG 생성 체인: 문서를 바탕으로 답변을 생성합니다.
    # - 입력: 질문 및 문서 컨텍스트
    # - 출력: 생성된 텍스트(최대 3문장의 간결한 답변)
    rag_chain = create_rag_chain(llm)

    # 환각 평가 체인: 생성된 답변이 문서에 근거하는지 평가합니다.
    # - 입력: 문서 및 생성된 답변
    # - 출력: {"score": "yes"} 또는 {"score": "no"} JSON
    # - 생성된 내용이 주어진 문서에 근거하면 yes 반환
    hallucination_grader = create_hallucination_grader(llm)

    # 답변 평가 체인: 생성된 답변이 질문에 유용한지 평가합니다.
    # - 입력: 질문 및 생성된 답변
    # - 출력: {"score": "yes"} 또는 {"score": "no"} JSON
    # - 생성된 답변이 질문 해결에 유용하면 yes 반환
    # answer_grader = create_answer_grader(llm)

    # 3. 워크플로우 구성
    # ----------------------------------------------------------------
    # 위에서 생성한 컴포넌트를 사용하여 전체 워크플로우 그래프를 구성합니다.
    # 워크플로우 구성:
    # - 노드: websearch(웹 검색), retrieve(문서 검색), grade_documents(문서 평가), generate(답변 생성)
    # - 조건부 진입점: 질문 라우터에 따라 websearch 또는 retrieve로 시작
    # - 엣지: 노드 간의 연결 및 조건부 흐름 설정
    app = build_workflow(
        tavily,  # 웹 검색 클라이언트
        retriever,  # 문서 검색기
        retrieval_grader,  # 검색 결과 평가기
        rag_chain,  # RAG 생성 체인
        hallucination_grader,  # 환각 평가기
        # answer_grader,  # 답변 평가기
        # question_router,  # 질문 라우터
    )

    # 4. 테스트 실행
    # ----------------------------------------------------------------
    # 구성된 워크플로우 그래프를 테스트 질문으로 실행합니다.
    # "What is prompt?"라는 질문에 대한 답변을 생성합니다.
    # 실행 과정:
    # 1. 질문 라우팅 → 2. 문서 검색 또는 웹 검색 → 3. 문서 평가 → 4. 답변 생성 → 5. 답변 평가
    # 다른 질문을 테스트하려면 이 파라미터를 변경하세요.
    run_test_query(app, "What is prompt?")


# 스크립트가 직접 실행될 때만 메인 함수 실행
# 다른 모듈에서 임포트할 때는 실행되지 않습니다.
if __name__ == "__main__":
    main()
