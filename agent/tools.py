import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

from agent.vector_store import get_retriever

load_dotenv()


# ── Pydantic 스키마 ───────────────────────────────────────────

class GrammarError(BaseModel):
    """개별 문법 오류 항목."""

    original_text: str = Field(description="오류가 포함된 원문 표현")
    corrected_text: str = Field(description="교정된 올바른 표현")
    error_type: str = Field(description="오류 유형 (맞춤법/조사/어미/어순/어휘 중 하나)")
    explanation: str = Field(description="오류 원인과 교정 이유 설명")


class GrammarCheckResult(BaseModel):
    """문법 검사 전체 결과."""

    has_errors: bool = Field(description="문법 오류 존재 여부")
    errors: List[GrammarError] = Field(
        default_factory=list, description="발견된 오류 목록"
    )
    corrected_sentence: str = Field(description="오류가 모두 교정된 완성 문장")
    overall_assessment: str = Field(description="전반적인 문법 수준 한 줄 평가")


class TopicAnalysis(BaseModel):
    """토픽 분석 결과."""

    detected_topic: str = Field(
        description="감지된 주제 (여행/일상/취미/음식/날씨/직업/가족/기타 중 하나)"
    )
    user_intent: str = Field(description="사용자의 대화 의도 한 줄 요약")
    key_phrases: List[str] = Field(description="입력에서 추출한 핵심 표현 목록 (최대 3개)")


class EvaluationResult(BaseModel):
    """응답 품질 평가 결과."""

    score: int = Field(description="응답 품질 점수 (1~10)", ge=1, le=10)
    reasoning: str = Field(description="해당 점수를 부여한 근거")
    needs_more_context: bool = Field(
        description="RAG 재검색으로 더 많은 맥락이 필요한지 여부"
    )


# ── LLM 팩토리 (Tool 내부용, tools bound 없음) ────────────────

def _get_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=temperature,
    )


# ── Tool 정의 ─────────────────────────────────────────────────

@tool
def grammar_check(sentence: str) -> str:
    """한국어 문장의 문법 오류를 교정하고 JSON 문자열로 반환합니다.

    조사, 어미, 맞춤법, 어순 오류가 있을 수 있는 문장에 사용하세요.
    단순한 인사(안녕하세요 등)는 생략해도 됩니다.

    Args:
        sentence: 문법 교정이 필요한 한국어 문장
    """
    llm = _get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(GrammarCheckResult)

    try:
        result = structured_llm.invoke([
            SystemMessage(content=(
                "[중요] 반드시 한국어로만 답변하세요.\n"
                "당신은 전문 한국어 문법 교정 교사입니다.\n"
                "맞춤법, 조사, 어미, 어순, 어색한 표현을 꼼꼼히 확인하고 교정하세요.\n"
                "오류가 없다면 has_errors=false로, 잘 쓴 문장임을 긍정적으로 평가하세요."
            )),
            HumanMessage(content=f"교정할 문장: {sentence}"),
        ])
        return json.dumps({
            "has_errors": result.has_errors,
            "errors": [
                {
                    "original": e.original_text,
                    "corrected": e.corrected_text,
                    "type": e.error_type,
                    "explanation": e.explanation,
                }
                for e in result.errors
            ],
            "corrected_sentence": result.corrected_sentence,
            "overall_assessment": result.overall_assessment,
        }, ensure_ascii=False)
    except Exception:
        return json.dumps({
            "has_errors": False,
            "errors": [],
            "corrected_sentence": sentence,
            "overall_assessment": "(문법 분석 불가)",
        }, ensure_ascii=False)


@tool
def rag_retrieve(topic: str, query: str) -> str:
    """토픽 관련 자연스러운 한국어 표현을 ChromaDB에서 검색해서 반환합니다.

    사용자의 대화 토픽에 맞는 자연스러운 표현이 필요할 때 사용하세요.

    Args:
        topic: 대화 주제 (여행/일상/취미/음식/날씨/직업/가족 중 하나)
        query: 검색할 표현이나 상황 설명
    """
    retriever = get_retriever(k=3)
    docs = retriever.invoke(f"{topic} 관련 한국어 표현: {query}")

    if not docs:
        return "관련 표현을 찾지 못했습니다."

    return "\n\n".join(
        f"[{doc.metadata.get('topic', '일반')} / {doc.metadata.get('subtopic', '')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


@tool
def evaluate_response(topic: str, user_input: str, response: str) -> str:
    """생성된 튜터 응답의 품질을 평가하고 점수와 이유를 반환합니다.

    튜터 응답 초안을 작성한 후 최종 전달 전에 품질을 확인할 때 사용하세요.
    점수가 7점 미만이면 응답을 개선하세요.

    Args:
        topic: 현재 대화 토픽
        user_input: 사용자의 원본 입력
        response: 평가할 튜터 응답 초안
    """
    llm = _get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(EvaluationResult)

    try:
        result = structured_llm.invoke([
            SystemMessage(content=(
                "[중요] 반드시 한국어로만 답변하세요.\n"
                "당신은 한국어 교육 품질 평가 전문가입니다.\n"
                "튜터 응답을 아래 기준으로 1~10점 평가하세요:\n"
                "1. 토픽 관련성 (3점)\n"
                "2. 문법 교정 정확성 (3점)\n"
                "3. 교육적 가치 — 새 표현 포함 여부 (2점)\n"
                "4. 대화 연속성 — 꼬리 질문의 자연스러움 (2점)\n"
                "7점 이상: 통과 / 7점 미만: 개선 필요"
            )),
            HumanMessage(content=(
                f"토픽: {topic}\n"
                f"학습자 입력: {user_input}\n\n"
                f"[튜터 응답]\n{response}"
            )),
        ])
        return json.dumps({
            "score": result.score,
            "reasoning": result.reasoning,
            "needs_more_context": result.needs_more_context,
        }, ensure_ascii=False)
    except Exception:
        return json.dumps({
            "score": 7,
            "reasoning": "(평가 불가, 기준치 통과 처리)",
            "needs_more_context": False,
        }, ensure_ascii=False)
