"""
한국어 회화 튜터링 에이전트 — 터미널 실행 스크립트.

사용법:
  python main.py

명령어:
  quit   종료
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agent.graph import build_graph  # noqa: E402

SEP = "-" * 60


def main() -> None:
    """MemorySaver로 대화 히스토리를 유지하는 메인 루프."""
    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "session"}}

    print(SEP)
    print("  한국어 회화 튜터링 에이전트")
    print("  종료: quit")
    print(SEP)
    print()

    while True:
        try:
            user_input = input("나: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n학습 수고하셨습니다!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "종료"):
            print("학습 수고하셨습니다!")
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # 마지막 AIMessage (tool_calls 없는 최종 응답)
        final = result["messages"][-1]
        print(f"\n튜터: {final.content}\n")


if __name__ == "__main__":
    main()
