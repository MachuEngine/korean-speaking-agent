from typing import TypedDict


class AgentState(TypedDict):
    """한국어 회화 튜터링 에이전트의 공유 상태."""

    user_input: str        # 사용자 텍스트 입력
    current_topic: str     # 현재 대화 토픽 (여행, 일상, 취미 등)
    grammar_feedback: dict # 문법 교정 결과 (오류 목록, 교정 문장 등)
    retrieved_context: str # RAG로 검색된 토픽 관련 어휘 및 예문
    final_response: str    # 튜터가 사용자에게 전달하는 최종 응답
    evaluation_score: int  # 응답 적절성 점수 (1~10, 루프 분기 기준)
    retry_count: int       # rag_retrieve_node 누적 실행 횟수 (무한 루프 방지)
    messages: list         # 대화 히스토리 누적 ({"role": ..., "content": ...})
    next_action: str       # LLM이 결정한 다음 노드 이름
