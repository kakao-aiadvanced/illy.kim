# RAG (Retrieval Augmented Generation) 시스템

이 프로젝트는 질문 응답 작업을 위한 RAG(Retrieval Augmented Generation) 시스템을 구현합니다.

## 주요 기능

1. **문서 검색**: 관련 문서를 검색하고 평가합니다.
2. **답변 생성**: 검색된 문서를 기반으로 답변을 생성합니다.
3. **품질 평가**: 생성된 답변의 품질을 평가합니다.

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:

`.env` 파일을 생성하고 다음과 같이 API 키를 설정합니다:

```
TAVILY_API_KEY=your_tavily_api_key_here
```

## 프로젝트 구조

프로젝트는 다음과 같은 모듈로 구성되어 있습니다:

- `models.py`: GraphState 타입 정의
- `initialization.py`: 초기화 함수 (Tavily, LLM, 벡터 스토어)
- `chains.py`: 체인 생성 함수 (라우터, 평가기 등)
- `nodes.py`: 노드 함수 (문서 검색, 답변 생성 등)
- `edges.py`: 엣지 함수 (질문 라우팅, 생성 결정 등)
- `workflow.py`: 워크플로우 구성 함수
- `main.py`: 메인 실행 함수

## 실행 방법

다음 명령어를 사용하여 시스템을 실행합니다:

```bash
streamlit run app.py 
```