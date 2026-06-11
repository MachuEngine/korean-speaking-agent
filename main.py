"""
한국어 회화 튜터링 에이전트 - 터미널 실행 스크립트.

사용법:
  python main.py

명령어:
  /topic    대화 토픽 변경
  quit      종료
"""

from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph  # noqa: E402 (dotenv must load first)
from agent.state import AgentState  # noqa: E402

TOPICS = ["여행", "일상", "취미", "음식", "날씨", "직업", "가족"]
SEP_WIDE = "=" * 60
SEP_THIN = "-" * 60


def _select_topic() -> str:
    """사용자가 대화 토픽을 선택하도록 안내하고 선택한 토픽을 반환."""
    print(SEP_WIDE)
    print("  한국어 회화 튜터링 에이전트에 오신 것을 환영합니다!")
    print(SEP_WIDE)
    print("\n대화 토픽을 선택하세요:\n")
    for i, topic in enumerate(TOPICS, 1):
        print(f"  {i}. {topic}")
    print("  0. 직접 입력\n")

    while True:
        raw = input("번호를 입력하세요 (기본값: 1): ").strip()
        if not raw:
            return TOPICS[0]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(TOPICS):
                return TOPICS[idx - 1]
            if idx == 0:
                custom = input("토픽을 직접 입력하세요: ").strip()
                return custom if custom else TOPICS[0]
        print("  올바른 번호를 입력해 주세요.")


def _print_result(result: dict) -> None:
    """에이전트 실행 결과를 터미널에 출력."""
    print()
    print(SEP_THIN)
    print(f"  [토픽: {result.get('current_topic', '?')}]")
    print(SEP_THIN)

    # 문법 교정이 있으면 먼저 표시
    grammar = result.get("grammar_feedback", {})
    if grammar.get("has_errors") and grammar.get("errors"):
        print("\n[문법 교정]")
        for err in grammar["errors"]:
            print(f"  · '{err['original']}' → '{err['corrected']}' ({err['type']})")
            print(f"    {err['explanation']}")
        corrected = grammar.get("corrected_sentence", "")
        if corrected:
            print(f"\n  교정된 문장: {corrected}")

    # 튜터 최종 응답
    print("\n[튜터]")
    print(result.get("final_response", "응답을 생성하지 못했습니다."))

    # 평가 점수 (디버그 정보)
    score = result.get("evaluation_score", 0)
    retries = result.get("retry_count", 0)
    print(f"\n  (응답 품질 점수: {score}/10 | RAG 검색 횟수: {retries})")
    print(SEP_THIN)


def main() -> None:
    """메인 실행 루프: 토픽 선택 후 사용자 입력을 반복 처리."""
    graph = build_graph()
    current_topic = _select_topic()

    print(f"\n토픽 '{current_topic}'(으)로 대화를 시작합니다!")
    print("명령어: /topic (토픽 변경) | quit (종료)\n")

    while True:
        try:
            user_input = input("나: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n학습 수고하셨습니다! 다음에 또 만나요!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "종료"):
            print("\n학습 수고하셨습니다! 다음에 또 만나요!")
            break

        if user_input == "/topic":
            current_topic = _select_topic()
            print(f"\n토픽이 '{current_topic}'(으)로 변경되었습니다.\n")
            continue

        # 초기 상태 구성
        initial_state: AgentState = {
            "user_input": user_input,
            "current_topic": current_topic,
            "grammar_feedback": {},
            "retrieved_context": "",
            "final_response": "",
            "evaluation_score": 0,
            "retry_count": 0,
        }

        print("\n[처리 중...]\n")

        result = graph.invoke(initial_state)

        # 분석 후 토픽이 변경되었을 수 있으므로 동기화
        current_topic = result.get("current_topic", current_topic)

        _print_result(result)
        print()


if __name__ == "__main__":
    main()
