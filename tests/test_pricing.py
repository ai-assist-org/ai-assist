"""Tests for the pricing module"""

import logging

from ai_assist.pricing import MODEL_PRICING, compute_turn_cost, get_pricing


class TestGetPricing:
    def test_exact_model_match(self):
        result = get_pricing("claude-opus-4-6")
        assert result == (5.00, 25.00, 6.25, 0.50)

    def test_dated_suffix_stripped(self):
        result = get_pricing("claude-opus-4-6-20260205")
        assert result == (5.00, 25.00, 6.25, 0.50)

    def test_vertex_at_suffix_stripped(self):
        result = get_pricing("claude-sonnet-4-6@20260219")
        assert result == (3.00, 15.00, 3.75, 0.30)

    def test_vertex_default_suffix_stripped(self):
        result = get_pricing("claude-haiku-4-5@default")
        assert result == (1.00, 5.00, 1.25, 0.10)

    def test_fable_pricing(self):
        result = get_pricing("claude-fable-5")
        assert result == (10.00, 50.00, 12.50, 1.00)

    def test_unknown_model_falls_back(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = get_pricing("claude-unknown-99")
        assert result == MODEL_PRICING["claude-sonnet-4-6"]
        assert "Unknown model" in caplog.text

    def test_cache_write_is_1_25x_input(self):
        for _model, (inp, _out, cw, _cr) in MODEL_PRICING.items():
            assert abs(cw - inp * 1.25) < 0.001

    def test_cache_read_is_0_1x_input(self):
        for _model, (inp, _out, _cw, cr) in MODEL_PRICING.items():
            assert abs(cr - inp * 0.1) < 0.001


class TestComputeTurnCost:
    def test_basic_cost(self):
        entry = {"input_tokens": 1000, "output_tokens": 500}
        cost = compute_turn_cost("claude-sonnet-4-6", entry)
        expected = (1000 * 3.00 + 500 * 15.00) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_with_cache_tokens(self):
        entry = {
            "input_tokens": 10000,
            "output_tokens": 2000,
            "cache_creation_input_tokens": 3000,
            "cache_read_input_tokens": 5000,
        }
        cost = compute_turn_cost("claude-opus-4-6", entry)
        plain_input = 10000 - 5000 - 3000  # 2000
        expected = (plain_input * 5.00 + 2000 * 25.00 + 3000 * 6.25 + 5000 * 0.50) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_zero_tokens(self):
        entry = {"input_tokens": 0, "output_tokens": 0}
        assert compute_turn_cost("claude-opus-4-6", entry) == 0.0

    def test_missing_cache_fields(self):
        entry = {"input_tokens": 100, "output_tokens": 50}
        cost = compute_turn_cost("claude-haiku-4-5", entry)
        expected = (100 * 1.00 + 50 * 5.00) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_none_cache_fields(self):
        entry = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
        }
        cost = compute_turn_cost("claude-haiku-4-5", entry)
        expected = (100 * 1.00 + 50 * 5.00) / 1_000_000
        assert abs(cost - expected) < 1e-10
