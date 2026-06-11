from typing import List

from pydantic import BaseModel, Field


class GrammarError(BaseModel):
    """개별 문법 오류 항목."""

    original_text: str = Field(description="오류가 포함된 원문 표현")
    corrected_text: str = Field(description="교정된 올바른 표현")
    error_type: str = Field(description="오류 유형 (맞춤법/조사/어미/어순/어휘 중 하나)")
    explanation: str = Field(description="오류 원인과 교정 이유 설명")


class GrammarCheckResult(BaseModel):
    """문법 검사 전체 결과 (grammar_check_tool_node 출력 스키마)."""

    has_errors: bool = Field(description="문법 오류 존재 여부")
    errors: List[GrammarError] = Field(
        default_factory=list, description="발견된 오류 목록"
    )
    corrected_sentence: str = Field(description="오류가 모두 교정된 완성 문장")
    overall_assessment: str = Field(description="전반적인 문법 수준 한 줄 평가")


class TopicAnalysis(BaseModel):
    """토픽 분석 결과 (analyze_input_node 출력 스키마)."""

    detected_topic: str = Field(
        description="감지된 주제 (여행/일상/취미/음식/날씨/직업/가족/기타 중 하나)"
    )
    user_intent: str = Field(description="사용자의 대화 의도 한 줄 요약")
    key_phrases: List[str] = Field(description="입력에서 추출한 핵심 표현 목록 (최대 3개)")


class EvaluationResult(BaseModel):
    """응답 품질 평가 결과 (evaluate_response_node 출력 스키마)."""

    score: int = Field(description="응답 품질 점수 (1~10)", ge=1, le=10)
    reasoning: str = Field(description="해당 점수를 부여한 근거")
    needs_more_context: bool = Field(
        description="RAG 재검색으로 더 많은 맥락이 필요한지 여부"
    )
