"""
LangGraph StateGraph 구성 모듈.

그래프 구조:
  START
    │
    ▼
  analyze_input          ← 토픽·의도 분석
    │
    ▼
  grammar_check          ← 문법 오류 교정 (Tool Calling)
    │
    ▼
  rag_retrieve ◄─────────────────────────────┐
    │                                         │
    ▼                                   (score < 7 & retry < MAX)
  generate_response                           │
    │                                         │
    ▼                                         │
  evaluate_response ──(score >= 7)──► END     │
    │                                         │
    └────────────────(score < 7)──────────────┘
"""

import os

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    analyze_input_node,
    evaluate_response_node,
    generate_response_node,
    grammar_check_tool_node,
    rag_retrieve_node,
)
from agent.state import AgentState

load_dotenv()

# 환경 변수에서 품질 기준값 로드 (기본: 7점 / 최대 재시도: 3회)
QUALITY_THRESHOLD: int = int(os.getenv("QUALITY_THRESHOLD", "7"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))


def _route_after_evaluation(state: AgentState) -> str:
    """
    Conditional Edge 라우팅 함수.

    - 평가 점수가 기준치 미만이고 재시도 횟수가 MAX_RETRIES 미만이면
      "retry" → rag_retrieve_node로 돌아가 다른 키워드로 재검색.
    - 그 외에는 "end" → 그래프 종료.
    """
    score = state.get("evaluation_score", 0)
    retry_count = state.get("retry_count", 0)

    if score < QUALITY_THRESHOLD and retry_count < MAX_RETRIES:
        print(
            f"  [평가] {score}/10점 — 기준 미달, "
            f"RAG 재검색 ({retry_count}/{MAX_RETRIES}회차)"
        )
        return "retry"

    print(f"  [평가] {score}/10점 — 기준 통과, 응답 완료")
    return "end"


def build_graph():
    """
    StateGraph를 구성하고 컴파일한 실행 가능한 그래프를 반환.

    노드 등록 → 선형 엣지 연결 → 조건부 엣지(루프) 설정 → 컴파일
    """
    graph = StateGraph(AgentState)

    # ── 노드 등록 ──────────────────────────────────────────────────
    graph.add_node("analyze_input", analyze_input_node)
    graph.add_node("grammar_check", grammar_check_tool_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("evaluate_response", evaluate_response_node)

    # ── 선형 엣지 (고정 순서) ──────────────────────────────────────
    graph.add_edge(START, "analyze_input")
    graph.add_edge("analyze_input", "grammar_check")
    graph.add_edge("grammar_check", "rag_retrieve")
    graph.add_edge("rag_retrieve", "generate_response")
    graph.add_edge("generate_response", "evaluate_response")

    # ── 조건부 엣지 (Self-Reflection 루프) ────────────────────────
    # evaluate_response 결과에 따라 rag_retrieve로 루프백하거나 종료
    graph.add_conditional_edges(
        "evaluate_response",
        _route_after_evaluation,
        {
            "retry": "rag_retrieve",  # 품질 미달 → RAG 재검색
            "end": END,               # 품질 통과 → 종료
        },
    )

    return graph.compile()
