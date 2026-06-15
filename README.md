# 한국어 회화 튜터링 파이프라인

LangGraph 기반의 한국어 회화 튜터링 시스템입니다.  
사용자의 한국어 입력을 받아 **문법을 교정**하고 **토픽에 맞는 대화**를 이어나갑니다.

> **이 시스템은 에이전트인가?**  
> 엄밀히는 **LLM 라우팅이 추가된 파이프라인**입니다.  
> `decide_next` 노드에서 LLM이 실행 경로를 한 번 결정하고, 이후 흐름은 고정된 순서로 실행됩니다.  
> 진짜 에이전트라면 LLM이 매 스텝마다 "다음에 뭘 할지"를 반복적으로 결정해야 합니다 (ReAct 패턴 등).

---

## 주요 기능

- **LLM 판단 라우팅**: 사용자 입력에 따라 문법 교정 / RAG 검색 / 즉시 응답 중 경로를 선택
- **문법 교정**: 조사·어미·맞춤법·어순 오류를 감지하고 교정 이유를 설명
- **RAG 기반 표현 추천**: ChromaDB에서 토픽별 한국어 표현을 검색해 응답에 반영
- **Self-Reflection 루프**: 생성된 응답을 스스로 평가(1~10점)해 기준 미달 시 재검색·재생성
- **꼬리 질문**: 대화가 이어지도록 매 응답 끝에 후속 질문 포함
- **토픽 자동 전환**: 입력 내용에 따라 대화 주제를 자동 감지·전환

지원 토픽: `여행` `일상` `취미` `음식` `날씨` `직업` `가족`

---

## 기술 스택

| 구성 요소 | 사용 기술 |
|---|---|
| LLM | OpenAI API — GPT-4o |
| LLM 클라이언트 | `langchain-openai` (ChatOpenAI) |
| 파이프라인 | LangGraph `StateGraph` |
| 임베딩 | `langchain-openai` OpenAIEmbeddings — text-embedding-3-small |
| 벡터 스토어 | ChromaDB (로컬 파일 영속화, 싱글턴 캐싱) |
| 환경 변수 | python-dotenv |

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
    ├── vector_store.py    # ChromaDB 초기화 및 샘플 데이터 로드 (싱글턴)
    ├── nodes.py           # 6개 노드 함수
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
    messages: list         # 대화 히스토리 누적
    next_action: str       # LLM이 결정한 다음 노드 이름
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

- 초기 실행 시 토픽별 한국어 표현 **19개**를 자동 삽입
- `get_retriever(k)` 함수로 유사도 기반 상위 k개 문서를 검색
- 재시도 횟수가 늘어날수록 k 값을 증가시켜 더 넓은 범위를 검색
- 모듈 수준 싱글턴(`_vector_store`)으로 프로세스 내 재생성 방지

### `agent/nodes.py` — 6개 노드

| 노드 | 역할 | LLM 호출 방식 |
|---|---|---|
| `analyze_input_node` | 토픽·의도 분석 | `with_structured_output(TopicAnalysis)` |
| `decide_next_node` | 다음 실행 노드 결정 (LLM 라우터) | `llm.invoke()` |
| `grammar_check_tool_node` | 문법 오류 교정 | `with_structured_output(GrammarCheckResult)` |
| `rag_retrieve_node` | ChromaDB 검색 | LLM 미사용 |
| `generate_response_node` | 튜터 응답 생성 + 대화 히스토리 누적 | `llm.invoke()` |
| `evaluate_response_node` | 응답 품질 평가 | `with_structured_output(EvaluationResult)` |

모든 `with_structured_output` 호출은 `try/except`로 감싸져 있어, 모델이 잘못된 JSON을 반환해도 기본값으로 폴백합니다.

### `agent/graph.py` — LangGraph 파이프라인

```
START
  │
  ▼
analyze_input        ← 토픽·의도 분석
  │
  ▼
decide_next          ← LLM이 다음 경로를 한 번 결정 (라우터)
  │
  ├─(grammar_check)──► grammar_check ──► rag_retrieve ──► generate_response
  │                                                              ▲
  ├─(rag_retrieve)──────────────────────► rag_retrieve ─────────┤
  │                                                              │
  └─(generate_response)──────────────────────────────────────────┘
                                                                 │
                                                                 ▼
                                                        evaluate_response
                                                                 │
                                            (score ≥ 7)──► END
                                            (score < 7 & retry < MAX)──► rag_retrieve
```

**결정 지점은 두 곳입니다:**

1. `decide_next`: LLM이 `next_action`을 결정 → 3가지 경로 중 하나로 분기
   - `grammar_check` → 문법 교정 후 RAG 검색 → 응답 생성
   - `rag_retrieve` → RAG 검색 → 응답 생성
   - `generate_response` → 바로 응답 생성

2. `evaluate_response`: 품질 점수 기반 Self-Reflection 루프
   - score ≥ 7 또는 retry ≥ MAX → `END`
   - score < 7 and retry < MAX → `rag_retrieve`로 루프백

**한계:** `decide_next` 이후 각 경로의 실행 순서는 그래프에 고정되어 있습니다. LLM은 경로 진입 시점만 결정할 뿐, 중간에 추가 판단을 하지 않습니다.

---

## 설치 및 실행

### 1. 프로젝트 설정

```bash
git clone https://github.com/MachuEngine/korean-speaking-agent.git
cd korean-speaking-agent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
```

### 2. 실행

```bash
python main.py
```

실행 후 토픽을 선택하고 한국어로 대화하면 됩니다.  
`/topic` 입력 시 토픽 변경, `quit` 입력 시 종료합니다.

---

## 환경 변수

`.env.example`을 복사하여 `.env`로 저장한 뒤 필요에 따라 수정합니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | (필수) | OpenAI API 키 |
| `OPENAI_MODEL` | `gpt-4o` | 사용할 LLM 모델 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB 저장 경로 |
| `QUALITY_THRESHOLD` | `7` | 응답 통과 기준 점수 |
| `MAX_RETRIES` | `3` | RAG 재검색 최대 횟수 |

---

## 알려진 제한 사항

- **에이전트가 아닌 파이프라인**: LLM의 판단은 `decide_next` 한 지점에서만 이루어지고 이후 흐름은 고정됩니다. 진짜 에이전트로 발전시키려면 LLM이 루프 안에서 매 스텝 도구를 선택하는 구조(ReAct 등)가 필요합니다.
- **대화 히스토리 미활용**: `messages`에 대화가 누적되지만 현재 노드들은 이를 프롬프트에 포함하지 않습니다.
- **구조화 출력 신뢰성**: 모델이 잘못된 JSON을 반환하면 폴백 기본값으로 처리됩니다.
- **라우터 판단 불확실성**: `decide_next`가 항상 최적 경로를 선택하지 않을 수 있으며, 판단 불확실 시 `grammar_check`(기본값)를 사용합니다.
