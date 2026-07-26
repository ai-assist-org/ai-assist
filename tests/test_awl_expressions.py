"""Tests for AWL expression evaluator"""

import pytest

from ai_assist.awl_expressions import AWLExpressionEvaluator


@pytest.fixture
def evaluator():
    return AWLExpressionEvaluator()


class TestIsTruthy:
    def test_none_is_falsy(self, evaluator):
        assert evaluator.is_truthy(None) is False

    def test_empty_string_is_falsy(self, evaluator):
        assert evaluator.is_truthy("") is False

    def test_empty_list_is_falsy(self, evaluator):
        assert evaluator.is_truthy([]) is False

    def test_zero_is_falsy(self, evaluator):
        assert evaluator.is_truthy(0) is False

    def test_non_empty_string_is_truthy(self, evaluator):
        assert evaluator.is_truthy("hello") is True

    def test_non_empty_list_is_truthy(self, evaluator):
        assert evaluator.is_truthy(["a"]) is True

    def test_non_zero_is_truthy(self, evaluator):
        assert evaluator.is_truthy(42) is True

    def test_dict_is_truthy(self, evaluator):
        assert evaluator.is_truthy({"key": "val"}) is True

    def test_empty_dict_is_falsy(self, evaluator):
        assert evaluator.is_truthy({}) is False


class TestEvaluate:
    def test_simple_variable_lookup(self, evaluator):
        result = evaluator.evaluate("handlers", {"handlers": ["a", "b"]})
        assert result == ["a", "b"]

    def test_missing_variable_returns_none(self, evaluator):
        result = evaluator.evaluate("missing", {})
        assert result is None

    def test_len_function(self, evaluator):
        result = evaluator.evaluate("len(handlers)", {"handlers": ["a", "b", "c"]})
        assert result == 3

    def test_len_empty(self, evaluator):
        result = evaluator.evaluate("len(items)", {"items": []})
        assert result == 0

    def test_comparison_greater(self, evaluator):
        result = evaluator.evaluate("len(handlers) > 0", {"handlers": ["a"]})
        assert result is True

    def test_comparison_greater_false(self, evaluator):
        result = evaluator.evaluate("len(handlers) > 0", {"handlers": []})
        assert result is False

    def test_comparison_equals(self, evaluator):
        result = evaluator.evaluate("count == 5", {"count": 5})
        assert result is True

    def test_comparison_not_equals(self, evaluator):
        result = evaluator.evaluate("status != failed", {"status": "ok"})
        assert result is True

    def test_comparison_less_than(self, evaluator):
        result = evaluator.evaluate("count < 10", {"count": 3})
        assert result is True

    def test_comparison_greater_equal(self, evaluator):
        result = evaluator.evaluate("count >= 5", {"count": 5})
        assert result is True

    def test_comparison_less_equal(self, evaluator):
        result = evaluator.evaluate("count <= 5", {"count": 5})
        assert result is True

    def test_not_variable_true(self, evaluator):
        result = evaluator.evaluate("not report_exists", {"report_exists": True})
        assert result is False

    def test_not_variable_false(self, evaluator):
        result = evaluator.evaluate("not report_exists", {"report_exists": False})
        assert result is True

    def test_not_missing_variable(self, evaluator):
        result = evaluator.evaluate("not report_exists", {})
        assert result is True

    def test_not_empty_list(self, evaluator):
        result = evaluator.evaluate("not items", {"items": []})
        assert result is True

    def test_not_non_empty_list(self, evaluator):
        result = evaluator.evaluate("not items", {"items": ["a"]})
        assert result is False

    def test_property_access(self, evaluator):
        result = evaluator.evaluate("config.entrypoint", {"config": {"entrypoint": "main.go"}})
        assert result == "main.go"

    def test_index_access(self, evaluator):
        result = evaluator.evaluate("handlers[0]", {"handlers": ["first", "second"]})
        assert result == "first"

    def test_comparison_with_none_left(self, evaluator):
        """None > 0 should not crash, should return False."""
        result = evaluator.evaluate("count > 0", {})
        assert result is False

    def test_comparison_with_none_right(self, evaluator):
        result = evaluator.evaluate("0 > count", {})
        assert result is False

    def test_index_out_of_range(self, evaluator):
        result = evaluator.evaluate("items[99]", {"items": ["a"]})
        assert result is None


class TestInterpolate:
    def test_simple_variable(self, evaluator):
        result = evaluator.interpolate("Find ${target}.", {"target": "HTTP server"})
        assert result == "Find HTTP server."

    def test_multiple_variables(self, evaluator):
        result = evaluator.interpolate("${a} and ${b}", {"a": "foo", "b": "bar"})
        assert result == "foo and bar"

    def test_no_variables(self, evaluator):
        result = evaluator.interpolate("No variables here.", {})
        assert result == "No variables here."

    def test_missing_variable_left_as_is(self, evaluator):
        result = evaluator.interpolate("Find ${missing}.", {})
        assert result == "Find ${missing}."

    def test_property_interpolation(self, evaluator):
        result = evaluator.interpolate("File: ${config.file}", {"config": {"file": "main.go"}})
        assert result == "File: main.go"


class TestValidateExpression:
    def test_valid_variable(self, evaluator):
        evaluator.validate_expression("handlers")

    def test_valid_len(self, evaluator):
        evaluator.validate_expression("len(handlers)")

    def test_valid_comparison(self, evaluator):
        evaluator.validate_expression("len(handlers) > 0")

    def test_valid_not(self, evaluator):
        evaluator.validate_expression("not report_exists")

    def test_valid_property_access(self, evaluator):
        evaluator.validate_expression("config.key")

    def test_valid_index_access(self, evaluator):
        evaluator.validate_expression("handlers[0]")

    def test_invalid_special_chars(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.validate_expression("!!!")

    def test_invalid_unclosed_len(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.validate_expression("len(")

    def test_invalid_empty(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.validate_expression("")

    def test_invalid_comparison_missing_right(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.validate_expression("handlers >")

    def test_invalid_comparison_missing_left(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.validate_expression("> 0")


class TestExtractVariables:
    def test_simple_variable(self, evaluator):
        assert evaluator.extract_variables("handlers") == {"handlers"}

    def test_len_function(self, evaluator):
        assert evaluator.extract_variables("len(items)") == {"items"}

    def test_comparison(self, evaluator):
        assert evaluator.extract_variables("count > 0") == {"count"}

    def test_comparison_two_variables(self, evaluator):
        assert evaluator.extract_variables("left == right") == {"left", "right"}

    def test_len_comparison(self, evaluator):
        assert evaluator.extract_variables("len(jobs) > 0") == {"jobs"}

    def test_not_expression(self, evaluator):
        assert evaluator.extract_variables("not done") == {"done"}

    def test_property_access(self, evaluator):
        assert evaluator.extract_variables("config.key") == {"config"}

    def test_index_access(self, evaluator):
        assert evaluator.extract_variables("items[0]") == {"items"}

    def test_numeric_literal_not_variable(self, evaluator):
        assert evaluator.extract_variables("len(items) > 0") == {"items"}

    def test_no_variables_in_numeric_comparison(self, evaluator):
        assert evaluator.extract_variables("5 > 3") == set()


class TestStripQuotes:
    """Test that agent-produced quoted values are handled correctly"""

    def test_single_quoted_number(self, evaluator):
        result = evaluator.evaluate("count > 0", {"count": "'5'"})
        assert result is True

    def test_single_quoted_zero(self, evaluator):
        result = evaluator.evaluate("count > 0", {"count": "'0'"})
        assert result is False

    def test_double_quoted_number(self, evaluator):
        result = evaluator.evaluate("count > 0", {"count": '"5"'})
        assert result is True

    def test_unquoted_string_number(self, evaluator):
        result = evaluator.evaluate("count > 0", {"count": "5"})
        assert result is True

    def test_plain_string_not_stripped(self, evaluator):
        """Non-quoted strings should not be altered."""
        result = evaluator.evaluate("status", {"status": "running"})
        assert result == "running"


class TestJsonStringHandling:
    """Test that JSON strings are parsed for len() and loop operations"""

    def test_len_json_string_array(self, evaluator):
        """len() on a JSON string array returns item count, not char count"""
        result = evaluator.evaluate("len(items)", {"items": '[{"id": 1}, {"id": 2}, {"id": 3}]'})
        assert result == 3

    def test_len_json_string_empty_array(self, evaluator):
        result = evaluator.evaluate("len(items)", {"items": "[]"})
        assert result == 0

    def test_len_json_string_comparison(self, evaluator):
        """len(items) > 0 works with JSON string arrays"""
        result = evaluator.evaluate("len(items) > 0", {"items": '[{"id": 1}]'})
        assert result is True

    def test_len_plain_string_returns_char_count(self, evaluator):
        """len() on a non-JSON string still returns character count"""
        result = evaluator.evaluate("len(name)", {"name": "hello"})
        assert result == 5

    def test_len_real_list(self, evaluator):
        """len() on a real Python list works as before"""
        result = evaluator.evaluate("len(items)", {"items": [1, 2, 3]})
        assert result == 3

    def test_len_python_repr_string(self, evaluator):
        """len() on a Python-repr string with single quotes"""
        result = evaluator.evaluate("len(items)", {"items": "[{'id': 'a'}, {'id': 'b'}]"})
        assert result == 2

    def test_len_single_dict_string(self, evaluator):
        """len() on a single dict string returns key count"""
        result = evaluator.evaluate("len(item)", {"item": "{'id': 'abc', 'name': 'test'}"})
        assert result == 2


class TestStringLiterals:
    def test_equality_single_quotes(self, evaluator):
        result = evaluator.evaluate("status == 'unknown'", {"status": "unknown"})
        assert result is True

    def test_equality_single_quotes_false(self, evaluator):
        result = evaluator.evaluate("status == 'unknown'", {"status": "ready"})
        assert result is False

    def test_equality_double_quotes(self, evaluator):
        result = evaluator.evaluate('status == "unknown"', {"status": "unknown"})
        assert result is True

    def test_not_equals_string(self, evaluator):
        result = evaluator.evaluate("status != 'done'", {"status": "running"})
        assert result is True

    def test_string_with_operator_chars(self, evaluator):
        """String literals containing operator characters don't break parsing"""
        result = evaluator.evaluate("msg == 'a>=b'", {"msg": "a>=b"})
        assert result is True

    def test_validate_string_literal(self, evaluator):
        evaluator.validate_expression("status == 'unknown'")

    def test_validate_double_quoted(self, evaluator):
        evaluator.validate_expression('status == "running"')

    def test_extract_variables_ignores_string(self, evaluator):
        result = evaluator.extract_variables("status == 'unknown'")
        assert result == {"status"}

    def test_smart_quotes_normalized(self, evaluator):
        result = evaluator.evaluate("status == ‘unknown’", {"status": "unknown"})
        assert result is True

    def test_bool_not_coerced_to_string(self, evaluator):
        """Boolean True should not match string 'unknown' via bool('unknown') coercion"""
        result = evaluator.evaluate("branch_ready == 'unknown'", {"branch_ready": True})
        assert result is False

    def test_unterminated_string_raises(self, evaluator):
        with pytest.raises(ValueError, match="Unterminated string"):
            evaluator.validate_expression("status == 'oops")
