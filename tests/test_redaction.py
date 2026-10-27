"""Tests for secret redaction before content reaches the LLM."""

import json

import pytest

from ai_assist.redaction import Redactor, filter_sensitive_env


class TestKnownValues:
    def test_redacts_sensitive_env_value_with_label(self):
        redactor = Redactor.from_environment({"GITHUB_TOKEN": "abcdef123456", "HOME": "/home/user"})

        assert redactor.redact("token=abcdef123456 home=/home/user") == (
            "token=[REDACTED:GITHUB_TOKEN] home=/home/user"
        )

    def test_short_values_are_not_redacted(self):
        redactor = Redactor.from_environment({"SOME_TOKEN": "true"})

        assert redactor.redact("value is true") == "value is true"

    def test_non_secret_config_names_are_not_redacted(self):
        redactor = Redactor.from_environment({"JIRA_URL": "https://issues.example.com"})

        assert redactor.redact("see https://issues.example.com/X-1") == "see https://issues.example.com/X-1"

    def test_longer_value_wins_over_contained_shorter_value(self):
        redactor = Redactor.from_environment({"A_TOKEN": "secretvalue", "B_TOKEN": "secretvalue-extended"})

        assert redactor.redact("secretvalue-extended") == "[REDACTED:B_TOKEN]"


class TestPatterns:
    @pytest.mark.parametrize(
        "secret",
        [
            "sk-ant-api03-" + "a" * 40,
            "sk-proj-" + "Ab1_-" * 8,
            "ghp_" + "a1B2" * 9,
            "github_pat_" + "a1B2_" * 10,
            "glpat-" + "a1B2c3D4e5F6g7H8i9J0",
            "xoxb-1234567890-abcdefghij",
            "AIza" + "B" * 35,
            "AKIAABCDEFGHIJKLMNOP",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl",
        ],
    )
    def test_known_formats_are_redacted(self, secret):
        redactor = Redactor.from_environment({})

        result = redactor.redact(f"value: {secret} end")

        assert secret not in result
        assert "[REDACTED]" in result
        assert result.endswith(" end")

    def test_bearer_token_redacted(self):
        result = Redactor.from_environment({}).redact("Authorization: Bearer 0123456789abcdef0123")

        assert "0123456789abcdef0123" not in result

    def test_url_credentials_redacted(self):
        result = Redactor.from_environment({}).redact("https://user:hunter2pass@example.com/repo")

        assert "hunter2pass" not in result
        assert "example.com/repo" in result

    def test_private_key_block_redacted(self):
        key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\nabc\n-----END RSA PRIVATE KEY-----"

        result = Redactor.from_environment({}).redact(f"before\n{key}\nafter")

        assert "MIIEow" not in result
        assert result.startswith("before\n") and result.endswith("\nafter")

    def test_git_sha_and_uuid_untouched(self):
        text = "commit 3f786850e387550fdab836ed7e6dc881de23001b id 123e4567-e89b-12d3-a456-426614174000"

        assert Redactor.from_environment({}).redact(text) == text


class TestUserConfig:
    def test_explicit_strings_and_patterns(self, tmp_path):
        config = tmp_path / "redact.json"
        config.write_text(json.dumps({"strings": ["project-zeus"], "patterns": [r"EMP-\d{6}"]}))

        redactor = Redactor.from_environment({}, config_path=config)

        assert redactor.redact("project-zeus owner EMP-123456") == "[REDACTED] owner [REDACTED]"

    def test_invalid_pattern_is_skipped(self, tmp_path):
        config = tmp_path / "redact.json"
        config.write_text(json.dumps({"strings": ["confidential"], "patterns": ["(unclosed"]}))

        redactor = Redactor.from_environment({}, config_path=config)

        assert redactor.redact("confidential (unclosed") == "[REDACTED] (unclosed"

    def test_missing_or_invalid_config_is_ignored(self, tmp_path):
        bad = tmp_path / "redact.json"
        bad.write_text("not json")

        assert Redactor.from_environment({}, config_path=bad).redact("plain") == "plain"
        assert Redactor.from_environment({}, config_path=tmp_path / "missing.json").redact("plain") == "plain"

    def test_default_config_path_is_in_config_dir(self, isolate_config_dir):
        (isolate_config_dir / "redact.json").write_text(json.dumps({"strings": ["internal-codename"]}))

        assert Redactor.from_environment({}).redact("internal-codename") == "[REDACTED]"


class TestFilterSensitiveEnv:
    def test_strips_secrets_keeps_rest(self):
        env = {"PATH": "/usr/bin", "GITHUB_TOKEN": "x", "ANTHROPIC_API_KEY": "y", "LANG": "C"}

        assert filter_sensitive_env(env) == {"PATH": "/usr/bin", "LANG": "C"}

    def test_allowlisted_names_kept(self):
        env = {"PATH": "/usr/bin", "GITHUB_TOKEN": "x", "JIRA_API_TOKEN": "y"}

        assert filter_sensitive_env(env, allowed={"GITHUB_TOKEN"}) == {"PATH": "/usr/bin", "GITHUB_TOKEN": "x"}
