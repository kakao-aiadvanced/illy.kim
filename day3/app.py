"""
RAG(Retrieval Augmented Generation) 시스템 Streamlit 애플리케이션

이 프로그램은 Streamlit을 사용하여 RAG 시스템을 시각화하고 사용자가 질문을 입력하면
이를 처리하여 결과를 보여줍니다.

주요 기능:
1. 사용자 질문 입력 받기
2. 질문 처리 과정 시각화 (라우팅, 문서 검색, 평가, 답변 생성)
3. 최종 생성된 답변 표시
4. 답변 한글 번역 기능
"""

import io
import streamlit as st
import time
from contextlib import redirect_stdout

# RAG 시스템 관련 모듈 임포트
from initialization import init_tavily_client, init_llm, setup_vectorstore
from chains import (
    create_retrieval_grader,
    create_rag_chain,
    create_hallucination_grader,
    # create_answer_grader,  # 제거됨
    # create_question_router,  # 주석 처리 (제거됨)
)
from workflow import build_workflow


def initialize_rag_system():
    """
    RAG 시스템의 모든 컴포넌트를 초기화하고 워크플로우를 구성합니다.

    Returns:
        app: 컴파일된 워크플로우 애플리케이션
        llm: 초기화된 LLM 모델 (번역에 사용)
    """
    # 진행 상태 표시를 위한 Streamlit 컴포넌트
    status_text = st.empty()
    progress_bar = st.progress(0)

    # 1. 초기화 (25%)
    status_text.text("Tavily 클라이언트 초기화 중...")
    tavily = init_tavily_client()
    progress_bar.progress(10)

    status_text.text("LLM 모델 초기화 중...")
    llm = init_llm()
    progress_bar.progress(20)

    status_text.text("벡터 스토어 설정 중...")
    retriever = setup_vectorstore()
    progress_bar.progress(30)

    # 2. 체인 생성 (65%)
    # 주석 처리: 질문 라우터 체인은 더 이상 사용하지 않음
    # status_text.text("질문 라우터 체인 생성 중...")
    # question_router = create_question_router(llm)
    # progress_bar.progress(35)

    status_text.text("검색 결과 평가 체인 생성 중...")
    retrieval_grader = create_retrieval_grader(llm)
    progress_bar.progress(45)

    status_text.text("RAG 생성 체인 생성 중...")
    rag_chain = create_rag_chain(llm)
    progress_bar.progress(55)

    status_text.text("환각 평가 체인 생성 중...")
    hallucination_grader = create_hallucination_grader(llm)
    progress_bar.progress(65)

    # 3. 워크플로우 구성 (100%)
    status_text.text("워크플로우 구성 중...")
    app = build_workflow(
        tavily,
        retriever,
        retrieval_grader,
        rag_chain,
        hallucination_grader,
        # question_router,  # 주석 처리 (제거됨)
    )
    progress_bar.progress(100)
    status_text.text("초기화 완료!")

    # 초기화 완료 후 진행 표시 제거
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

    return app, llm


def capture_output_and_run_query(app, question):
    """
    실행 과정의 출력을 캡처하면서 질문을 처리합니다.

    Args:
        app: 컴파일된 워크플로우 애플리케이션
        question: 사용자 질문

    Returns:
        tuple: (결과 텍스트, 세부 출력 로그)
    """
    # 출력을 캡처하기 위한 StringIO 객체
    f = io.StringIO()

    # 출력 리디렉션
    with redirect_stdout(f):
        # workflow.py의 run_test_query 함수는 question 파라미터가 없음
        # 대신 inputs을 직접 설정하고 실행
        from workflow import run_test_query

        # 원래 run_test_query 함수를 오버라이드하기 위한 내부 함수
        def run_with_question(app, question_text):
            # 입력 정의: 질문을 포함하는 딕셔너리
            inputs = {"question": question_text}

            # 애플리케이션을 스트리밍 모드로 실행하고, 각 출력에 대해 반복합니다.
            last_value = None
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(
                        f"Finished running: {key}:"
                    )  # 각 단계가 완료될 때마다 출력합니다.
                    last_value = value

            # 최종 생성된 답변을 출력하고 반환합니다.
            if "generation" in last_value:
                print(f"Final answer: {last_value['generation']}")
                return last_value["generation"]
            else:
                print("No generation found in the final state.")
                return "답변을 생성하지 못했습니다."

        result = run_with_question(app, question)

    # 캡처된 출력 가져오기
    output_log = f.getvalue()

    return result, output_log


def translate_to_korean(llm, text):
    """
    영어 텍스트를 한글로 번역합니다.

    Args:
        llm: 초기화된 LLM 모델
        text: 번역할 영어 텍스트

    Returns:
        str: 한글로 번역된 텍스트
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # 번역 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 전문 번역가입니다. 다음 영어 텍스트를 한국어로 정확하게 번역하세요. 번역문은 자연스러운 한국어로 작성해주세요.",
            ),
            ("human", "{text}"),
        ]
    )

    # 번역 체인 생성
    translation_chain = prompt | llm | StrOutputParser()

    # 번역 수행
    korean_text = translation_chain.invoke({"text": text})
    return korean_text


def main():
    """
    Streamlit 애플리케이션의 메인 함수
    """
    # 앱 제목 및 설명
    st.title("RAG 시스템 - 질문 답변 데모")
    st.markdown(
        """
    이 데모는 RAG(Retrieval Augmented Generation) 시스템을 시각화합니다.
    질문을 입력하면, 시스템은 다음 단계로 처리합니다:
    
    1. **문서 검색**: 벡터 저장소에서 관련 문서 검색
    2. **관련성 평가**: 검색된 문서의 관련성 평가 (필요시 웹 검색)
    3. **답변 생성**: 검색된 문서를 기반으로 답변 생성
    4. **품질 평가**: 생성된 답변의 품질 평가
    """
    )

    # 세션 상태를 사용하여 앱 초기화 상태 유지
    if "app" not in st.session_state or "llm" not in st.session_state:
        with st.spinner("RAG 시스템 초기화 중..."):
            st.session_state.app, st.session_state.llm = initialize_rag_system()

    # 질문 입력 필드
    question = st.text_input("질문을 입력하세요:", value="What is prompt?")

    # 실행 버튼
    if st.button("답변 생성"):
        with st.spinner("질문 처리 중..."):
            # 결과 생성 및 로그 캡처
            result, output_log = capture_output_and_run_query(
                st.session_state.app, question
            )

            # 결과 탭으로 구성
            tab1, tab2, tab3 = st.tabs(["답변 (영어)", "답변 (한글)", "처리 과정"])

            # 영어 답변 탭
            with tab1:
                st.success("생성된 답변:")
                st.write(result)

            # 한글 번역 탭
            with tab2:
                with st.spinner("한글로 번역 중..."):
                    korean_result = translate_to_korean(st.session_state.llm, result)
                    st.success("한글 번역:")
                    st.write(korean_result)

            # 처리 과정 탭
            with tab3:
                st.text("RAG 시스템 처리 과정 로그:")
                st.code(output_log, language="text")


if __name__ == "__main__":
    main()
