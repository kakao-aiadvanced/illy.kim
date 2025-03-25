import streamlit as st
import json
import requests

st.set_page_config(
    page_title="Gemma 챗봇",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("💭 Gemma3 12b")

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    temperature = st.slider("Temperature (창의성 조절)", 0.0, 2.0, 0.7, 0.1)
    memory_length = st.slider("메모리 길이 (이전 대화 수)", 0, 10, 3, 1)
    system_prompt = st.text_area(
        "시스템 프롬프트",
        value="당신은 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 친절하고 정확하게 답변해주세요.",
        height=100,
    )

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def create_prompt_with_memory(new_prompt, memory_length):
    # 시스템 프롬프트 추가
    full_prompt = f"{system_prompt}\n\n"

    # 이전 대화 내용 추가
    if memory_length > 0:
        memory_messages = st.session_state.messages[
            -memory_length * 2 :
        ]  # 각 대화는 질문과 답변 2개의 메시지를 가짐
        for msg in memory_messages:
            role_prefix = "사용자: " if msg["role"] == "user" else "어시스턴트: "
            full_prompt += f"{role_prefix}{msg['content']}\n"

    # 새로운 프롬프트 추가
    full_prompt += f"사용자: {new_prompt}\n어시스턴트: "

    return full_prompt


# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 표시
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Ollama API 호출
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # 메모리를 포함한 프롬프트 생성
            full_prompt = create_prompt_with_memory(prompt, memory_length)

            # 스트리밍 응답 처리
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3:12b",
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "stream": True,
                },
                stream=True,
            )

            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    response_part = json_response.get("response", "")
                    full_response += response_part
                    # 실시간으로 응답 업데이트
                    message_placeholder.markdown(full_response + "▌")

            # 최종 응답 표시 (커서 제거)
            message_placeholder.markdown(full_response)
            # 메시지 저장
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        except Exception as e:
            message_placeholder.error(f"오류 발생: {str(e)}")

# 채팅 초기화 버튼
if st.sidebar.button("대화 초기화"):
    st.session_state.messages = []
    st.rerun()
