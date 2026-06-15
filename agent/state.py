from langgraph.graph import MessagesState

# Tool 기반 에이전트 상태.
# messages 필드 하나만 사용하며 add_messages 리듀서로 자동 누적됨.
# 기존의 grammar_feedback, retrieved_context 등 별도 필드는 불필요:
# 모든 Tool 결과가 ToolMessage.content 형태로 messages 안에서 전달됨.
AgentState = MessagesState
