"""AWL expression evaluator and variable interpolation"""

import ast
import json
import re
from typing import Any


def _parse_str_to_collection(val: str) -> Any:
    """Try to parse a string as a JSON or Python collection."""
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list | dict):
            return parsed
    except json.JSONDecodeError, ValueError:
        pass
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list | dict):
            return parsed
    except ValueError, SyntaxError:
        pass
    return val


_TWO_CHAR_OPS = frozenset({">=", "<=", "!=", "=="})
_ONE_CHAR_OPS = frozenset({">", "<"})
_OP_CHARS = frozenset({"!", ">", "<", "="})

NOT = "NOT"
OP = "OP"
STR = "STR"
VALUE = "VALUE"


_SMART_QUOTE_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",  # smart single quotes
        "“": '"',
        "”": '"',  # smart double quotes
    }
)


def _tokenize(expression: str) -> list[tuple[str, str]]:
    """Tokenize an AWL expression into (type, value) pairs."""
    expression = expression.translate(_SMART_QUOTE_MAP)
    tokens: list[tuple[str, str]] = []
    i = 0
    n = len(expression)

    while i < n:
        if expression[i].isspace():
            i += 1
            continue

        # Quoted string literal
        if expression[i] in ("'", '"'):
            quote = expression[i]
            j = i + 1
            while j < n and expression[j] != quote:
                j += 1
            if j >= n:
                raise ValueError(f"Unterminated string literal starting at position {i}")
            tokens.append((STR, expression[i : j + 1]))
            i = j + 1
            continue

        # Two-char operator
        if i + 1 < n and expression[i : i + 2] in _TWO_CHAR_OPS:
            tokens.append((OP, expression[i : i + 2]))
            i += 2
            continue

        # One-char operator
        if expression[i] in _ONE_CHAR_OPS:
            tokens.append((OP, expression[i]))
            i += 1
            continue

        # Value token: consume until whitespace, operator char, or quote
        j = i
        while (
            j < n and not expression[j].isspace() and expression[j] not in _OP_CHARS and expression[j] not in ("'", '"')
        ):
            j += 1
        if j == i:
            # Unrecognized char (e.g. bare '=' or '!') — consume it so we don't loop forever
            j += 1
        tokens.append((VALUE, expression[i:j]))
        i = j

    # Convert leading "not" value to NOT token
    if tokens and tokens[0] == (VALUE, "not"):
        tokens[0] = (NOT, "not")

    return tokens


class AWLExpressionEvaluator:
    def is_truthy(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            # Treat agent-produced boolean strings as their boolean equivalents
            if value.strip().lower() in ("false", "0", "null", "none", ""):
                return False
            return True
        if isinstance(value, list | dict):
            return len(value) > 0
        if isinstance(value, int | float):
            return value != 0
        return bool(value)

    _OPS: dict[str, Any] = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def evaluate(self, expression: str, variables: dict[str, Any]) -> Any:
        tokens = _tokenize(expression.strip())

        if not tokens:
            return None

        # not <sub-expression>
        if tokens[0][0] == NOT:
            rest = expression.strip()[4:].strip()
            return not self.is_truthy(self.evaluate(rest, variables))

        # <value> <op> <value>
        if len(tokens) == 3 and tokens[1][0] == OP:
            left_val = self._resolve_value(tokens[0], variables)
            right_val = self._resolve_value(tokens[2], variables)
            op = tokens[1][1]

            if isinstance(left_val, int | float) and not isinstance(left_val, bool) and isinstance(right_val, str):
                try:
                    right_val = type(left_val)(right_val)
                except ValueError, TypeError:
                    pass
            elif isinstance(right_val, int | float) and not isinstance(right_val, bool) and isinstance(left_val, str):
                try:
                    left_val = type(right_val)(left_val)
                except ValueError, TypeError:
                    pass

            if left_val is None or right_val is None:
                if op in ("==", "!="):
                    return self._OPS[op](left_val, right_val)
                return False

            return self._OPS[op](left_val, right_val)

        # Single value
        if len(tokens) == 1:
            return self._resolve_value(tokens[0], variables)

        return None

    @staticmethod
    def _strip_quotes(val: Any) -> Any:
        """Strip surrounding quotes from string values produced by agents."""
        if isinstance(val, str):
            stripped = val.strip()
            if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in ("'", '"'):
                return stripped[1:-1]
        return val

    def _resolve_value(self, token: tuple[str, str], variables: dict[str, Any]) -> Any:  # noqa: PLR0911
        tok_type, expr = token

        # String literal — return inner content
        if tok_type == STR:
            return expr[1:-1]

        expr = expr.strip()

        len_match = re.match(r"len\((\w+)\)", expr)
        if len_match:
            var_name = len_match.group(1)
            val = variables.get(var_name)
            if val is None:
                return 0
            # Parse JSON or Python-repr strings that are actually arrays/objects
            if isinstance(val, str):
                val = _parse_str_to_collection(val)
            return len(val)

        for num_type in (int, float):
            try:
                return num_type(expr)
            except ValueError:
                pass

        index_match = re.match(r"(\w+)\[(\d+)\]", expr)
        if index_match:
            var_name = index_match.group(1)
            index = int(index_match.group(2))
            val = variables.get(var_name)
            if val is not None and isinstance(val, list | tuple):
                if index < len(val):
                    return val[index]
            return None

        if "." in expr:
            parts = expr.split(".", 1)
            val = variables.get(parts[0])
            if isinstance(val, dict):
                return val.get(parts[1])
            return None

        return self._strip_quotes(variables.get(expr))

    # Valid token: variable, len(var), var[N], var.prop, numeric literal, or string literal
    _TOKEN_RE = re.compile(r"^(len\(\w+\)|\w+\[\d+\]|\w+\.\w+|\w+|\d+(\.\d+)?|'[^']*'|\"[^\"]*\")$")

    def _validate_token(self, token: tuple[str, str]) -> None:
        if token[0] == STR:
            return
        if not self._TOKEN_RE.match(token[1]):
            raise ValueError(f"Invalid expression token: '{token[1]}'")

    def validate_expression(self, expression: str) -> None:
        expression = expression.strip()
        if not expression:
            raise ValueError("Empty expression")

        tokens = _tokenize(expression)

        # not <sub-expression>
        if tokens and tokens[0][0] == NOT:
            rest = expression[4:].strip()
            self.validate_expression(rest)
            return

        # <value> <op> <value>
        if len(tokens) == 3 and tokens[1][0] == OP:
            self._validate_token(tokens[0])
            self._validate_token(tokens[2])
            return

        # Single value
        if len(tokens) == 1:
            self._validate_token(tokens[0])
            return

        raise ValueError(f"Invalid expression: '{expression}'")

    def extract_variables(self, expression: str) -> set[str]:
        """Extract variable names referenced in an expression."""
        expression = expression.strip()
        tokens = _tokenize(expression)
        variables: set[str] = set()

        if not tokens:
            return variables

        # not <sub-expression>
        if tokens[0][0] == NOT:
            return self.extract_variables(expression[4:].strip())

        for token in tokens:
            if token[0] == VALUE:
                variables |= self._extract_token_variable(token[1])

        return variables

    def _extract_token_variable(self, token: str) -> set[str]:
        """Extract the variable name from a single token."""
        token = token.strip()

        # len(var)
        len_match = re.match(r"len\((\w+)\)", token)
        if len_match:
            return {len_match.group(1)}

        # Numeric literal — not a variable
        try:
            int(token)
            return set()
        except ValueError:
            pass
        try:
            float(token)
            return set()
        except ValueError:
            pass

        # var[N] or var.prop
        index_match = re.match(r"(\w+)\[\d+\]", token)
        if index_match:
            return {index_match.group(1)}

        if "." in token:
            return {token.split(".", 1)[0]}

        # Plain variable name
        if re.match(r"^\w+$", token):
            return {token}

        return set()

    def interpolate(self, text: str, variables: dict[str, Any]) -> str:
        def replacer(match: re.Match) -> str:
            expr = match.group(1)
            val = self._resolve_value((VALUE, expr), variables)
            if val is None:
                return match.group(0)
            return str(val)

        return re.sub(r"\$\{([^}]+)\}", replacer, text)
