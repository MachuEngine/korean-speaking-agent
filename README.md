# 한국어 회화 튜터링 에이전트

LangGraph 기반의 순환형 AI 에이전트로, 사용자의 한국어 입력을 받아 **문법을 교정**하고 **토픽에 맞는 대화**를 이어나갑니다.

---

## 주요 기능

- **문법 교정**: 조사·어미·맞춤법·어순 오류를 감지하고 교정 이유를 설명
- **RAG 기반 표현 추천**: ChromaDB에서 토픽별 자연스러운 한국어 표현 검색 후 응답에 반영
- **Self-Reflection 루프**: 생성된 응답을 스스로 평가(1~10점)하여 기준 미달 시 재검색·재생성
- **꼬리 질문**: 대화가 자연스럽게 이어지도록 매 응답 끝에 후속 질문 포함
- **토픽 자동 전환**: 입력 내용에 따라 대화 주제를 자동으로 감지·전환

지원 토픽: `여행` `일상` `취미` `음식` `날씨` `직업` `가족`

---

## 기술 스택

| 구성 요소 | 사용 기술 |
|---|---|
| LLM | Ollama 로컬 서빙 — Qwen 2.5 7B Instruct |
| LLM 클라이언트 | `langchain-openai` (ChatOpenAI, Ollama OpenAI 호환 엔드포인트) |
| 파이프라인 | LangGraph `StateGraph` |
| 임베딩 | `langchain-ollama` OllamaEmbeddings — nomic-embed-text |
| 벡터 스토어 | ChromaDB (로컬 파일 영속화) |
| 환경 변수 | python-dotenv |

> LLM을 `ChatOpenAI`로 추상화했기 때문에 `base_url`과 `model`만 변경하면 OpenAI, Anthropic 등 클라우드 모델로 전환 가능합니다.

---

## 코드 구조

```
korean-speaking-agent/
├── main.py                # 터미널 인터랙티브 실행 스크립트
├── requirements.txt
├── .env.example           # 환경 변수 템플릿
└── agent/
    ├── state.py           # AgentState (TypedDict) 정의
    ├── tools.py           # Pydantic 출력 스키마 3종
    ├── vector_store.py    # ChromaDB 초기화 및 샘플 데이터 로드
    ├── nodes.py           # 5개 노드 함수
    └── graph.py           # StateGraph 구성 및 컴파일
```

### `agent/state.py` — 공유 상태

```python
class AgentState(TypedDict):
    user_input: str        # 사용자 입력
    current_topic: str     # 현재 대화 토픽
    grammar_feedback: dict # 문법 교정 결과
    retrieved_context: str # RAG 검색 결과
    final_response: str    # 튜터 최종 응답
    evaluation_score: int  # 응답 품질 점수 (1~10)
    retry_count: int       # RAG 재검색 횟수 (무한 루프 방지)
```

그래프의 모든 노드가 이 상태를 공유하며, 각 노드는 자신이 변경한 키만 반환합니다.

### `agent/tools.py` — 구조화 출력 스키마

LLM의 `with_structured_output()`에 사용되는 Pydantic 모델 3종입니다.

| 클래스 | 사용 노드 | 역할 |
|---|---|---|
| `TopicAnalysis` | `analyze_input_node` | 감지된 토픽, 사용자 의도, 핵심 표현 |
| `GrammarCheckResult` | `grammar_check_tool_node` | 오류 목록, 교정 문장, 전체 평가 |
| `EvaluationResult` | `evaluate_response_node` | 품질 점수, 평가 근거 |

### `agent/vector_store.py` — ChromaDB

- 초기 실행 시 토픽별 한국어 표현 **19개**를 자동으로 삽입
- `get_retriever(k)` 함수로 유사도 기반 상위 k개 문서를 검색
- 재시도 횟수가 늘어날수록 k 값을 증가시켜 더 넓은 범위를 검색

### `agent/nodes.py` — 5개 노드

| 노드 | 역할 | LLM 호출 방식 |
|---|---|---|
| `analyze_input_node` | 토픽·의도 분석 | `with_structured_output(TopicAnalysis)` |
| `grammar_check_tool_node` | 문법 오류 교정 | `with_structured_output(GrammarCheckResult)` |
| `rag_retrieve_node` | ChromaDB 검색 | LLM 미사용 |
| `generate_response_node` | 튜터 응답 생성 | `llm.invoke()` |
| `evaluate_response_node` | 응답 품질 평가 | `with_structured_output(EvaluationResult)` |

모든 `with_structured_output` 호출은 `try/except`로 감싸져 있어, 로컬 모델이 잘못된 JSON을 반환해도 기본값으로 폴백하여 그래프가 중단되지 않습니다.

### `agent/graph.py` — LangGraph 파이프라인

```
START
  │
  ▼
analyze_input       ← 토픽·의도 분석
  │
  ▼
grammar_check       ← 문법 교정 (Tool Calling)
  │
  ▼
rag_retrieve ◄──────────────────────────────────┐
  │                                              │
  ▼                                        (score < 7
generate_response                          & retry < 3)
  │                                              │
  ▼                                              │
evaluate_response ──(score ≥ 7)──► END          │
  │                                              │
  └────────────────────────────────────────────►┘
```

`evaluate_response` 이후 조건부 엣지(`add_conditional_edges`)가 점수와 재시도 횟수를 확인합니다.
- **score ≥ 7 또는 retry ≥ 3** → `END`
- **score < 7 and retry < 3** → `rag_retrieve`로 루프백

---

## 설치 및 실행

### 1. 사전 요구사항

```bash
# Ollama 설치 (macOS)
brew install ollama

# 필요한 모델 pull
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

### 2. 프로젝트 설정

```bash
git clone https://github.com/MachuEngine/korean-speaking-agent.git
cd korean-speaking-agent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

### 3. 실행

```bash
# Ollama 서버 시작 (별도 터미널)
ollama serve

# 에이전트 실행
python main.py
```

실행 후 토픽을 선택하고 한국어로 대화하면 됩니다.  
`/topic` 입력 시 토픽 변경, `quit` 입력 시 종료합니다.

---

## 환경 변수

`.env.example`을 복사하여 `.env`로 저장한 뒤 필요에 따라 수정합니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI 호환 엔드포인트 |
| `OLLAMA_API_KEY` | `ollama` | 더미 API 키 |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | 사용할 LLM |
| `OLLAMA_EMBED_BASE_URL` | `http://localhost:11434` | 임베딩 엔드포인트 |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | 임베딩 모델 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB 저장 경로 |
| `QUALITY_THRESHOLD` | `7` | 응답 통과 기준 점수 |
| `MAX_RETRIES` | `3` | RAG 재검색 최대 횟수 |

---

## 테스트 결과

로컬 환경(CPU-only, Qwen 2.5 7B Instruct)에서 실행한 결과입니다.

### 테스트 1 — 문법 정상 문장 (음식 토픽)

```
입력:  저는 음식 먹기을 좋아해요. 특히 한국 음식은 맛있어요.
토픽:  음식
```

**문법 교정:**
- `저는 음식 먹기을 좋아해요.` → `저는 음식을 먹기 좋아해요.` (조사 오류)
- `특히 한국 음식은 맛있어요.` → `특히 한국 음식은 맛있습니다.` (어미)

**튜터 응답 (요약):**
> 안녕하세요! 한국 음식은 정말 다양하고 맛있어요. '먹기' 앞에 '을'을 추가하고, '맛있어요'를 '맛있습니다'로 바꾸시면 더 자연스러워요.  
> 다음에 어떤 음식을 좋아하시나요?

- 응답 품질 점수: **9/10**
- RAG 재검색 루프: **없음 (1회 통과)**

### 테스트 2 — Self-Reflection 루프 동작 확인 (여행 토픽)

```
입력:  저는 어제 비행기로 여행을 갔어요. 호텔에서 체크인하는게 복잡했어요.
토픽:  여행
```

**문법 교정:**
- `체크인하는게` → `체크인하기가` (조사 오류)

**루프 동작:**
```
1차 평가: 6/10 → 기준 미달, RAG 재검색 (1/3회차)
2차 평가: 8/10 → 기준 통과, 응답 완료
```

- Self-Reflection 루프 **정상 동작 확인**

---

## 알려진 제한 사항

- **CPU-only 환경에서 응답 속도가 느립니다.** 1턴당 5회 LLM 호출이 발생하며, GPU 없이 7B 모델을 실행하면 응답에 수 분이 소요될 수 있습니다.
- **Qwen 2.5의 중국어 혼용.** 프롬프트 첫 줄에 한국어 전용 지시를 배치하여 개선했으나, 간헐적으로 혼용될 수 있습니다.
- **구조화 출력 신뢰성.** 로컬 7B 모델이 잘못된 JSON을 반환하면 폴백 기본값으로 처리됩니다.
