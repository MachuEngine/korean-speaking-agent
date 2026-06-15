"""
Tool 단위 테스트.

- rag_retrieve: LLM 미사용 → ChromaDB 검색 구조 검증
- grammar_check: LLM 사용 → 반환 JSON 구조·타입 검증 (내용 비결정적이므로 검증 안 함)
- evaluate_response: LLM 사용 → 반환 JSON 구조·점수 범위 검증
"""

import json

import pytest

from agent.tools import evaluate_response, grammar_check, rag_retrieve


# ── rag_retrieve ──────────────────────────────────────────────

class TestRagRetrieve:
    def test_반환값이_문자열(self):
        result = rag_retrieve.invoke({"topic": "여행", "query": "공항 표현"})
        assert isinstance(result, str)

    def test_결과가_비어있지_않음(self):
        result = rag_retrieve.invoke({"topic": "음식", "query": "식당 표현"})
        assert len(result) > 0

    def test_없는_토픽도_문자열_반환(self):
        # ChromaDB에 없는 주제여도 폴백 문자열을 반환해야 함
        result = rag_retrieve.invoke({"topic": "알수없는토픽", "query": "알수없는쿼리"})
        assert isinstance(result, str)


# ── grammar_check ─────────────────────────────────────────────

class TestGrammarCheck:
    def test_유효한_JSON_반환(self):
        result = grammar_check.invoke({"sentence": "저는 학교에 가요."})
        data = json.loads(result)  # 파싱 실패 시 테스트 실패
        assert data is not None

    def test_필수_키_존재(self):
        result = grammar_check.invoke({"sentence": "저는 학교에 가요."})
        data = json.loads(result)
        assert "has_errors" in data
        assert "errors" in data
        assert "corrected_sentence" in data
        assert "overall_assessment" in data

    def test_has_errors_타입이_bool(self):
        result = grammar_check.invoke({"sentence": "저는 음식 먹기을 좋아해요."})
        data = json.loads(result)
        assert isinstance(data["has_errors"], bool)

    def test_errors_타입이_리스트(self):
        result = grammar_check.invoke({"sentence": "저는 학교에 가요."})
        data = json.loads(result)
        assert isinstance(data["errors"], list)

    def test_corrected_sentence_타입이_문자열(self):
        result = grammar_check.invoke({"sentence": "저는 학교에 가요."})
        data = json.loads(result)
        assert isinstance(data["corrected_sentence"], str)
        assert len(data["corrected_sentence"]) > 0

    def test_errors_항목_구조(self):
        # 오류가 있을 때 각 항목이 올바른 키를 가져야 함
        result = grammar_check.invoke({"sentence": "저는 음식 먹기을 좋아해요."})
        data = json.loads(result)
        for err in data["errors"]:
            assert "original" in err
            assert "corrected" in err
            assert "type" in err
            assert "explanation" in err


# ── evaluate_response ─────────────────────────────────────────

class TestEvaluateResponse:
    SAMPLE = {
        "topic": "일상",
        "user_input": "안녕하세요!",
        "response": "안녕하세요! 정말 잘하셨어요. '오늘 하루 어떻게 보내셨어요?'라는 표현도 써보세요. 오늘 기분은 어떠세요?",
    }

    def test_유효한_JSON_반환(self):
        result = evaluate_response.invoke(self.SAMPLE)
        data = json.loads(result)
        assert data is not None

    def test_필수_키_존재(self):
        result = evaluate_response.invoke(self.SAMPLE)
        data = json.loads(result)
        assert "score" in data
        assert "reasoning" in data
        assert "needs_more_context" in data

    def test_score_범위(self):
        result = evaluate_response.invoke(self.SAMPLE)
        data = json.loads(result)
        assert isinstance(data["score"], int)
        assert 1 <= data["score"] <= 10

    def test_needs_more_context_타입이_bool(self):
        result = evaluate_response.invoke(self.SAMPLE)
        data = json.loads(result)
        assert isinstance(data["needs_more_context"], bool)

    def test_reasoning_타입이_문자열(self):
        result = evaluate_response.invoke(self.SAMPLE)
        data = json.loads(result)
        assert isinstance(data["reasoning"], str)
        assert len(data["reasoning"]) > 0
