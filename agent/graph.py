"""
LangGraph Tool 기반 ReAct 에이전트.

그래프 구조:
  START → agent → (tool_calls 있으면) → tools → agent → ...
                  (tool_calls 없으면) → END
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import AgentState
from agent.tools import evaluate_response, grammar_check, rag_retrieve

load_dotenv()

SYSTEM_PROMPT = """당신은 친절하고 격려적인 한국어 회화 튜터입니다.
반드시 한국어로만 답변하세요. 영어, 중국어 등 다른 언어는 절대 사용하지 마세요.

사용자의 한국어 입력에 대해 아래 순서로 도구를 사용하세요:
1. grammar_check — 문법 오류 확인 (단순 인사는 생략 가능)
2. rag_retrieve — 토픽에 맞는 자연스러운 표현 검색
3. 수집한 정보를 바탕으로 친절한 튜터 응답 초안 작성
4. evaluate_response — 초안 품질 평가, 점수 7점 미만이면 개선 후 재평가
5. 최종 응답 전달

최종 응답 형식:
1. 학습자의 노력을 칭찬하는 한 마디
2. 문법 교정 설명 (오류가 있을 때만)
3. 토픽 관련 자연스러운 표현 1~2개 소개
4. 대화를 이어가는 꼬리 질문 하나

말투: 친근한 존댓말 (~해요, ~네요, ~어요)"""

TOOLS = [grammar_check, rag_retrieve, evaluate_response]


def build_graph(checkpointer=None):
    """ReAct 에이전트 그래프를 구성하고 컴파일해서 반환."""
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.7,
    ).bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        """시스템 프롬프트를 앞에 붙여 LLM을 호출한다."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
