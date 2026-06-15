"""
LangGraph StateGraph 구성 모듈.

그래프 구조:
  START
    │
    ▼
  analyze_input          ← 토픽·의도 분석
    │
    ▼
  decide_next            ← LLM 판단 라우터
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
                                              (score >= 7)──► END  │
                                              (score < 7 & retry < MAX)
                                                                   │
                                                          rag_retrieve (루프백)
"""

import os

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    analyze_input_node,
    decide_next_node,
    evaluate_response_node,
    generate_response_node,
    grammar_check_tool_node,
    rag_retrieve_node,
)
from agent.state import AgentState

load_dotenv()

def build_graph():
    """
    StateGraph를 구성하고 컴파일한 실행 가능한 그래프를 반환.

    노드 등록 → 엣지 연결 → 조건부 엣지(라우팅·루프) 설정 → 컴파일
    """
    quality_threshold = int(os.getenv("QUALITY_THRESHOLD", "7"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))

    def _route_after_decide(state: AgentState) -> str:
        """next_action 값을 읽어 다음 노드를 결정."""
        return state.get("next_action", "grammar_check")

    def _route_after_evaluation(state: AgentState) -> str:
        """품질 기준을 클로저로 캡처하여 build_graph() 시점의 설정을 사용."""
        score = state.get("evaluation_score", 0)
        retry_count = state.get("retry_count", 0)

        if score < quality_threshold and retry_count < max_retries:
            print(
                f"  [평가] {score}/10점 — 기준 미달, "
                f"RAG 재검색 ({retry_count}/{max_retries}회차)"
            )
            return "retry"

        print(f"  [평가] {score}/10점 — 기준 통과, 응답 완료")
        return "end"

    graph = StateGraph(AgentState)

    # ── 노드 등록 ──────────────────────────────────────────────────
    graph.add_node("analyze_input", analyze_input_node)
    graph.add_node("decide_next", decide_next_node)
    graph.add_node("grammar_check", grammar_check_tool_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("evaluate_response", evaluate_response_node)

    # ── 선형 엣지 ──────────────────────────────────────────────────
    graph.add_edge(START, "analyze_input")
    graph.add_edge("analyze_input", "decide_next")
    graph.add_edge("grammar_check", "rag_retrieve")
    graph.add_edge("rag_retrieve", "generate_response")
    graph.add_edge("generate_response", "evaluate_response")

    # ── 조건부 엣지: LLM 판단 라우팅 ─────────────────────────────
    graph.add_conditional_edges(
        "decide_next",
        _route_after_decide,
        {
            "grammar_check": "grammar_check",
            "rag_retrieve": "rag_retrieve",
            "generate_response": "generate_response",
        },
    )

    # ── 조건부 엣지: Self-Reflection 루프 ─────────────────────────
    graph.add_conditional_edges(
        "evaluate_response",
        _route_after_evaluation,
        {
            "retry": "rag_retrieve",  # 품질 미달 → RAG 재검색
            "end": END,               # 품질 통과 → 종료
        },
    )

    return graph.compile()
