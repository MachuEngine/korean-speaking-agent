"""
그래프 통합 테스트.

- 컴파일: checkpointer 유무에 따른 빌드 성공 여부
- 노드 구조: agent / tools 노드 존재 확인
- 실행: messages 구조, 마지막 메시지 타입·내용 검증
"""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_graph


# ── 컴파일 ───────────────────────────────────────────────────

class TestBuildGraph:
    def test_checkpointer_없이_빌드(self):
        graph = build_graph()
        assert graph is not None

    def test_MemorySaver로_빌드(self):
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_agent_노드_존재(self):
        nodes = list(build_graph().get_graph().nodes.keys())
        assert "agent" in nodes

    def test_tools_노드_존재(self):
        nodes = list(build_graph().get_graph().nodes.keys())
        assert "tools" in nodes


# ── 실행 ─────────────────────────────────────────────────────

class TestGraphInvoke:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = build_graph(checkpointer=MemorySaver())
        self.config = {"configurable": {"thread_id": "test"}}

    def _invoke(self, text: str) -> dict:
        return self.graph.invoke(
            {"messages": [HumanMessage(content=text)]},
            config=self.config,
        )

    def test_messages_키_반환(self):
        result = self._invoke("안녕하세요!")
        assert "messages" in result

    def test_마지막_메시지에_content_존재(self):
        result = self._invoke("안녕하세요!")
        last = result["messages"][-1]
        assert isinstance(last.content, str)
        assert len(last.content) > 0

    def test_마지막_메시지가_AI_타입(self):
        result = self._invoke("안녕하세요!")
        last = result["messages"][-1]
        assert last.type == "ai"

    def test_마지막_메시지에_tool_calls_없음(self):
        # 에이전트가 tool_calls가 있는 메시지로 종료되면 안 됨
        result = self._invoke("안녕하세요!")
        last = result["messages"][-1]
        assert getattr(last, "tool_calls", []) == []

    def test_응답이_한국어_포함(self):
        result = self._invoke("안녕하세요!")
        last = result["messages"][-1]
        # 한국어 유니코드 범위(AC00-D7A3)의 문자가 하나 이상 있어야 함
        assert any("가" <= ch <= "힣" for ch in last.content)
