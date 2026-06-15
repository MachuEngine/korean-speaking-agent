"""
각 LangGraph 노드의 실행 로직을 담은 모듈.

노드 실행 순서:
  analyze_input → decide_next → (grammar_check →) rag_retrieve → generate_response → evaluate_response
                                  ↘ rag_retrieve ↗
                                  ↘ generate_response ↗
                └────────────────────────────────────────────────────────┘
                              (점수 미달 시 rag_retrieve로 루프백)
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.tools import EvaluationResult, GrammarCheckResult, TopicAnalysis
from agent.vector_store import get_retriever

load_dotenv()


# ──────────────────────────────────────────────
# LLM 팩토리
# ──────────────────────────────────────────────

def _get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """OpenAI 공식 엔드포인트를 사용하는 LLM 인스턴스를 반환."""
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=temperature,
    )


# ──────────────────────────────────────────────
# Node 1: analyze_input_node
# ──────────────────────────────────────────────

def analyze_input_node(state: AgentState) -> dict:
    """
    사용자 입력을 분석해 대화 토픽과 의도를 파악한다.

    - current_topic이 이미 설정되어 있으면 유지 (명시적 변경이 없는 한).
    - 처음이거나 '기타'로 감지되면 '일상'을 기본값으로 사용.
    반환: current_topic 업데이트
    """
    llm = _get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(TopicAnalysis)

    current_topic = state.get("current_topic", "")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """[중요] 당신은 반드시 한국어로만 답변해야 합니다. 영어, 중국어 등 다른 언어는 절대 사용하지 마세요.

당신은 한국어 회화 토픽 분석 전문가입니다.
사용자의 입력을 분석하여 가장 적합한 대화 주제를 선택하세요.

선택 가능한 토픽: 여행, 일상, 취미, 음식, 날씨, 직업, 가족, 기타
현재 설정된 토픽: {current_topic}

규칙:
- 현재 토픽과 관련된 내용이면 그 토픽을 유지하세요.
- 명확히 다른 주제로 전환되는 경우에만 새 토픽을 선택하세요.
- 현재 토픽이 없다면 입력에서 가장 적합한 토픽을 선택하세요.""",
        ),
        ("human", "사용자 입력: {user_input}"),
    ])

    try:
        result = structured_llm.invoke(
            prompt.format_messages(
                current_topic=current_topic or "미설정 (새로 감지 필요)",
                user_input=state["user_input"],
            )
        )
        new_topic = (
            result.detected_topic
            if result.detected_topic != "기타"
            else (current_topic or "일상")
        )
    except Exception:
        new_topic = current_topic or "일상"

    return {"current_topic": new_topic}


# ──────────────────────────────────────────────
# Node 2: decide_next_node
# ──────────────────────────────────────────────

def decide_next_node(state: AgentState) -> dict:
    """
    LLM이 사용자 입력을 보고 다음에 실행할 노드를 결정한다.

    판단 기준:
    - 문법 오류가 의심되면 → "grammar_check"
    - 토픽 관련 표현이 필요하면 → "rag_retrieve"
    - 간단한 인사나 단답이면 → "generate_response"
    - 판단이 어려우면 → "grammar_check" (기본값)
    반환: next_action (노드 이름)
    """
    llm = _get_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """[중요] 당신은 반드시 한국어로만 답변해야 합니다. 영어, 중국어 등 다른 언어는 절대 사용하지 마세요.

당신은 한국어 회화 튜터링 파이프라인의 라우터입니다.
사용자 입력을 분석하여 다음에 실행할 작업 이름을 결정하세요.

판단 기준:
- 문법 오류가 의심되면 → grammar_check
- 특정 토픽 관련 자연스러운 표현이 필요하면 → rag_retrieve
- 간단한 인사, 단답, 또는 문법·표현 지도가 불필요한 경우 → generate_response
- 판단이 어려우면 → grammar_check (기본값)

반드시 아래 세 값 중 하나만 정확히 출력하세요. 다른 텍스트는 포함하지 마세요:
grammar_check
rag_retrieve
generate_response""",
        ),
        ("human", "사용자 입력: {user_input}"),
    ])

    try:
        response = llm.invoke(
            prompt.format_messages(user_input=state["user_input"])
        )
        next_action = response.content.strip().strip('"').strip("'")
        if next_action not in ("grammar_check", "rag_retrieve", "generate_response"):
            next_action = "grammar_check"
    except Exception:
        next_action = "grammar_check"

    return {"next_action": next_action}


# ──────────────────────────────────────────────
# Node 3: grammar_check_tool_node
# ──────────────────────────────────────────────

def grammar_check_tool_node(state: AgentState) -> dict:
    """
    Tool Calling(structured output)을 활용해 사용자 문장의 문법 오류를 분석·교정한다.

    검사 항목: 맞춤법, 조사, 어미, 어순, 어색한 표현
    반환: grammar_feedback (오류 목록, 교정 문장, 전체 평가)
    """
    llm = _get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(GrammarCheckResult)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """[중요] 당신은 반드시 한국어로만 답변해야 합니다. 영어, 중국어 등 다른 언어는 절대 사용하지 마세요.

당신은 전문 한국어 문법 교정 교사입니다.
사용자의 한국어 문장을 꼼꼼히 분석하여 문법 오류를 찾고 교정해 주세요.

검사 항목:
1. 맞춤법 (띄어쓰기·철자)
2. 조사 (을/를, 이/가, 은/는, 에/에서 등)
3. 어미 (아/어, -았/-었, -(으)ㄹ 등)
4. 어순
5. 어색한 표현 (더 자연스러운 대안 제시)

오류가 없다면 has_errors=false로, 잘 쓴 문장임을 긍정적으로 평가하세요.""",
        ),
        ("human", "교정할 문장: {user_input}"),
    ])

    try:
        result = structured_llm.invoke(
            prompt.format_messages(user_input=state["user_input"])
        )
        grammar_feedback = {
            "has_errors": result.has_errors,
            "errors": [
                {
                    "original": err.original_text,
                    "corrected": err.corrected_text,
                    "type": err.error_type,
                    "explanation": err.explanation,
                }
                for err in result.errors
            ],
            "corrected_sentence": result.corrected_sentence,
            "overall_assessment": result.overall_assessment,
        }
    except Exception:
        grammar_feedback = {
            "has_errors": False,
            "errors": [],
            "corrected_sentence": state["user_input"],
            "overall_assessment": "(문법 분석 불가)",
        }

    return {"grammar_feedback": grammar_feedback}


# ──────────────────────────────────────────────
# Node 4: rag_retrieve_node
# ──────────────────────────────────────────────

def rag_retrieve_node(state: AgentState) -> dict:
    """
    ChromaDB에서 현재 토픽과 연관된 한국어 표현·예문을 검색한다.

    재시도 횟수(retry_count)에 따라 검색 쿼리와 문서 수(k)를 다양화하여
    매번 다른 맥락을 제공한다.
    반환: retrieved_context, retry_count(+1)
    """
    retry_count = state.get("retry_count", 0)
    topic = state.get("current_topic", "일상")
    user_input = state["user_input"]

    # 재시도 회차별 쿼리 전략: 좁은 → 넓은 범위로 확장
    if retry_count == 0:
        query = f"{topic} 관련 한국어 표현: {user_input}"
        k = 3
    elif retry_count == 1:
        query = f"{topic} 회화 예문 자연스러운 표현"
        k = 4
    else:
        query = "한국어 일상 회화 기본 표현"
        k = 5

    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)

    context = "\n\n".join(
        f"[{doc.metadata.get('topic', '일반')} / {doc.metadata.get('subtopic', '')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )

    return {
        "retrieved_context": context,
        "retry_count": retry_count + 1,
    }


# ──────────────────────────────────────────────
# Node 5: generate_response_node
# ──────────────────────────────────────────────

def generate_response_node(state: AgentState) -> dict:
    """
    문법 교정 결과와 RAG 맥락을 바탕으로 친절한 튜터 말투의 최종 응답을 생성한다.

    응답 구조:
      1. 학습자 노력 칭찬
      2. 문법 교정 내용 친절히 설명 (오류 있을 때)
      3. 토픽 관련 자연스러운 표현 1~2개 소개
      4. 대화를 이어가는 꼬리 질문 하나
    반환: final_response, messages(누적)
    """
    llm = _get_llm(temperature=0.7)

    grammar = state.get("grammar_feedback", {})
    topic = state.get("current_topic", "일상")
    context = state.get("retrieved_context", "")
    user_input = state["user_input"]

    # 문법 교정 섹션 포맷팅
    if grammar.get("has_errors") and grammar.get("errors"):
        error_lines = "\n".join(
            f"  · '{e['original']}' → '{e['corrected']}' ({e['type']}): {e['explanation']}"
            for e in grammar["errors"]
        )
        grammar_section = (
            f"[문법 교정 사항]\n{error_lines}\n"
            f"[교정된 문장] {grammar.get('corrected_sentence', user_input)}"
        )
    else:
        grammar_section = (
            f"[문법 평가] {grammar.get('overall_assessment', '잘 작성된 문장입니다!')}"
        )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """[중요] 당신은 반드시 한국어로만 답변해야 합니다. 영어, 중국어 등 다른 언어는 절대 사용하지 마세요.

당신은 친절하고 격려적인 한국어 회화 튜터입니다.
학습자가 자신감을 갖도록 항상 따뜻하고 긍정적인 말투로 피드백을 제공하세요.

응답 형식 (반드시 아래 순서로 작성):
1. 학습자의 노력을 칭찬하는 한 마디
2. 문법 교정 내용을 쉽게 설명 (오류가 있을 때만)
3. 현재 토픽과 관련된 자연스러운 한국어 표현 1~2개 소개
4. 대화를 이어나가는 꼬리 질문 하나

말투 규칙: 친근한 존댓말 (예: ~해요, ~네요, ~어요)
현재 대화 토픽: {topic}""",
        ),
        (
            "human",
            """학습자 입력: {user_input}

{grammar_section}

[RAG 검색 결과 - 토픽 관련 참고 표현]
{context}

위 내용을 바탕으로 친절한 튜터의 말투로 피드백과 꼬리 질문을 작성해 주세요.""",
        ),
    ])

    response = llm.invoke(
        prompt.format_messages(
            topic=topic,
            user_input=user_input,
            grammar_section=grammar_section,
            context=context,
        )
    )

    # 대화 히스토리 누적
    messages = state.get("messages", [])
    new_messages = messages + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response.content},
    ]

    return {"final_response": response.content, "messages": new_messages}


# ──────────────────────────────────────────────
# Node 6: evaluate_response_node
# ──────────────────────────────────────────────

def evaluate_response_node(state: AgentState) -> dict:
    """
    생성된 튜터 응답을 Self-Reflection 방식으로 평가한다.

    평가 기준 (총 10점):
      - 토픽 관련성   3점
      - 문법 교정 정확성  3점
      - 교육적 가치   2점  (새 표현·예문 포함 여부)
      - 대화 연속성   2점  (꼬리 질문의 자연스러움)

    반환: evaluation_score (1~10)
    """
    llm = _get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(EvaluationResult)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """[중요] 당신은 반드시 한국어로만 답변해야 합니다. 영어, 중국어 등 다른 언어는 절대 사용하지 마세요.

당신은 한국어 교육 품질 평가 전문가입니다.
튜터 응답이 아래 기준에 얼마나 부합하는지 1~10점으로 평가하세요.

평가 기준:
1. 토픽 관련성 (3점): 현재 대화 토픽과 관련이 있는가?
2. 문법 교정 정확성 (3점): 교정 내용이 올바르고 명확한가?
3. 교육적 가치 (2점): 새로운 표현이나 예문이 포함되어 있는가?
4. 대화 연속성 (2점): 꼬리 질문이 자연스럽게 대화를 이어가는가?

7점 이상: 품질 통과 / 7점 미만: 개선 또는 추가 맥락 필요""",
        ),
        (
            "human",
            """현재 토픽: {topic}
학습자 입력: {user_input}

[생성된 튜터 응답]
{final_response}

응답을 평가해 주세요.""",
        ),
    ])

    try:
        result = structured_llm.invoke(
            prompt.format_messages(
                topic=state.get("current_topic", "일상"),
                user_input=state["user_input"],
                final_response=state.get("final_response", ""),
            )
        )
        score = result.score
    except Exception:
        # 구조화 출력 실패 시 기준치(7)로 통과 처리하여 무한 루프 방지
        score = 7

    return {"evaluation_score": score}
