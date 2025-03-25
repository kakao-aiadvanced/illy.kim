# Gemma 챗봇

Streamlit으로 만든 Gemma 3 대화형 챗봇입니다.

## 필요 조건

- Python 3.8 이상
- Ollama가 설치되어 있어야 합니다
- Gemma 3 모델이 Ollama에 다운로드되어 있어야 합니다

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Gemma 3 모델 다운로드 (아직 설치하지 않은 경우):
```bash
ollama pull gemma3:12b
```

## 실행 방법

다음 명령어로 애플리케이션을 실행합니다:
```bash
streamlit run app.py
```

## 기능

- 실시간 대화형 인터페이스
- Temperature 조절을 통한 응답의 창의성 조절
- 대화 기록 저장
- 대화 초기화 기능
- 메모리 기능
  - 이전 대화 내용을 컨텍스트로 활용
  - 메모리 길이 조절 가능 (0-10개의 이전 대화)
  - 시스템 프롬프트 커스터마이징

## 주의사항

- Ollama가 실행 중이어야 합니다
- 로컬 서버(localhost:11434)에서 Ollama API를 사용합니다
- 메모리 길이를 늘리면 더 자연스러운 대화가 가능하지만, 토큰 사용량이 증가할 수 있습니다

# illy's AI-Advanced

## Day 1: LLM + Prompt Engineering

### 1. LLM

