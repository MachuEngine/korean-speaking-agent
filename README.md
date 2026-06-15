# 한국어 회화 튜터링 에이전트

LangGraph 기반의 Tool 사용 ReAct 에이전트입니다.  
사용자의 한국어 입력을 받아 **문법을 교정**하고 **토픽에 맞는 대화**를 이어나갑니다.

> **이전 버전과의 차이**  
> 이전: LLM이 `decide_next` 노드에서 경로를 한 번 선택하는 **파이프라인**  
> 현재: LLM이 Tool을 자율적으로 선택·실행하고 결과를 보고 다음 행동을 결정하는 **ReAct 에이전트**

---

## 동작 방식 (ReAct 루프)

```
START → agent → tool_calls 있으면 → tools → agent → ...
                tool_calls 없으면 → END
```

LLM은 매 스텝마다 아래 세 Tool 중 무엇을 쓸지 스스로 결정합니다:

| Tool | 역할 | 호출 시점 |
|---|---|---|
| `grammar_check` | 한국어 문장 문법 교정 | 문법 오류가 의심될 때 |
| `rag_retrieve` | ChromaDB에서 토픽별 표현 검색 | 자연스러운 표현이 필요할 때 |
| `evaluate_response` | 튜터 응답 초안 품질 평가 (1~10점) | 최종 응답 전달 전 자기 점검 |

전형적인 실행 흐름:
1. `grammar_check` 호출 → 오류 교정
2. `rag_retrieve` 호출 → 토픽 관련 표현 수집
3. LLM이 응답 초안 작성 (Tool 파라미터로 포함)
4. `evaluate_response` 호출 → 품질 점수 확인
5. 점수 ≥ 7이면 최종 응답 출력, 미만이면 개선 후 재평가

---

## 기술 스택

| 구성 요소 | 사용 기술 |
|---|---|
| LLM | OpenAI API — GPT-4o |
| 에이전트 패턴 | LangGraph ReAct (`ToolNode` + `tools_condition`) |
| 대화 히스토리 | LangGraph `MemorySaver` |
| 임베딩 | `langchain-openai` OpenAIEmbeddings — text-embedding-3-small |
| 벡터 스토어 | ChromaDB (로컬 파일 영속화, 싱글턴 캐싱) |
| 환경 변수 | python-dotenv |

---

## 코드 구조

```
korean-speaking-agent/
├── main.py                # 터미널 실행 스크립트 (MemorySaver 기반 루프)
├── requirements.txt
├── .env.example           # 환경 변수 템플릿
└── agent/
    ├── state.py           # AgentState = MessagesState (langgraph 내장)
    ├── tools.py           # Pydantic 스키마 + @tool 3개
    ├── vector_store.py    # ChromaDB 초기화 및 샘플 데이터 로드 (싱글턴)
    └── graph.py           # ReAct StateGraph 구성 및 컴파일
```

### `agent/state.py` — 상태

```python
from langgraph.graph import MessagesState
AgentState = MessagesState
```

`messages` 필드 하나만 사용합니다. `add_messages` 리듀서가 자동으로 누적 처리하며, 모든 Tool 결과는 `ToolMessage.content` 형태로 messages 안에서 전달됩니다.

### `agent/tools.py` — Tool 3개

각 Tool의 docstring을 LLM이 읽고 언제 어떤 Tool을 쓸지 판단합니다.

```python
@tool
def grammar_check(sentence: str) -> str:
    """한국어 문장의 문법 오류를 교정하고 JSON 문자열로 반환합니다."""

@tool
def rag_retrieve(topic: str, query: str) -> str:
    """토픽 관련 자연스러운 한국어 표현을 ChromaDB에서 검색해서 반환합니다."""

@tool
def evaluate_response(topic: str, user_input: str, response: str) -> str:
    """생성된 튜터 응답의 품질을 평가하고 점수와 이유를 반환합니다."""
```

`grammar_check`와 `evaluate_response`는 내부적으로 LLM을 사용하며, Tool 내부의 LLM은 에이전트 LLM과 독립적으로 동작합니다 (tools bind 없음).

### `agent/graph.py` — ReAct 그래프

```python
llm = ChatOpenAI(...).bind_tools(TOOLS)   # Tool 목록을 LLM에 연결

graph.add_node("agent", agent_node)        # LLM 호출 노드
graph.add_node("tools", ToolNode(TOOLS))  # Tool 실행 노드

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # tool_calls 여부로 분기
graph.add_edge("tools", "agent")                       # Tool 실행 후 agent로 복귀
```

`tools_condition`은 LLM 응답에 `tool_calls`가 있으면 `"tools"`, 없으면 `END`로 라우팅합니다.

---

## 설치 및 실행

```bash
git clone https://github.com/MachuEngine/korean-speaking-agent.git
cd korean-speaking-agent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

python main.py
```

종료: `quit`

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | (필수) | OpenAI API 키 |
| `OPENAI_MODEL` | `gpt-4o` | 사용할 LLM 모델 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB 저장 경로 |
| `QUALITY_THRESHOLD` | `7` | 응답 통과 기준 점수 (evaluate_response 참고용) |
| `MAX_RETRIES` | `3` | 에이전트가 참고하는 재시도 권장 횟수 |

---

## 이전 버전과의 구조 비교

| 항목 | 파이프라인 (이전) | ReAct 에이전트 (현재) |
|---|---|---|
| 실행 경로 | 그래프에 하드코딩 | LLM이 매 스텝 결정 |
| Tool 선택 | `decide_next` 노드에서 1회 | 루프 안에서 반복적으로 |
| 상태 | 7개 필드 (TypedDict) | `messages` 1개 (MessagesState) |
| 대화 히스토리 | 수동 누적 | MemorySaver 자동 관리 |
| 코드 파일 수 | 5개 (state/tools/nodes/graph/main) | 4개 (state/tools/graph/main) |

---

## 알려진 제한 사항

- **Tool 호출 순서 비보장**: LLM이 항상 grammar_check → rag_retrieve → evaluate_response 순서로 호출하지 않을 수 있습니다. 시스템 프롬프트로 권장 순서를 안내하지만 강제하지는 않습니다.
- **Tool 내부 LLM 호출 비용**: `grammar_check`와 `evaluate_response`가 각각 LLM을 호출하므로 한 턴에 최대 5회 LLM 호출이 발생할 수 있습니다.
