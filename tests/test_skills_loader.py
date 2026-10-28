"""Tests for skills loader"""

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_assist.skills_loader import SkillsLoader


def test_load_skill_from_local():
    """Test loading a skill from local directory"""
    loader = SkillsLoader()

    # Load test skill
    skill_path = Path("/tmp/test-skills/hello")
    content = loader.load_skill_from_local(skill_path)

    # Verify metadata
    assert content.metadata.name == "hello"
    assert "greets users" in content.metadata.description
    assert content.metadata.license == "MIT"
    assert content.metadata.source_type == "local"

    # Verify body
    assert "Hello Skill" in content.body
    assert "warmly and enthusiastically" in content.body


def test_invalid_skill_missing_file():
    """Test loading skill with missing SKILL.md"""
    loader = SkillsLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_skill_from_local(Path("/tmp/nonexistent"))


def test_skill_validation():
    """Test skill metadata validation"""
    from ai_assist.skills_loader import SkillMetadata

    # Valid skill
    valid = SkillMetadata(
        name="test-skill",
        description="A test skill",
        skill_path=Path("/tmp/test-skill"),
        source_type="local",
    )
    valid.validate()  # Should not raise

    # Invalid name (too long)
    invalid_long = SkillMetadata(
        name="a" * 65,  # 65 chars
        description="Test",
        skill_path=Path("/tmp/test"),
        source_type="local",
    )
    with pytest.raises(ValueError, match="must be 1-64 characters"):
        invalid_long.validate()

    # Invalid name (uppercase)
    invalid_upper = SkillMetadata(
        name="Test-Skill",  # Has uppercase
        description="Test",
        skill_path=Path("/tmp/test-skill"),
        source_type="local",
    )
    with pytest.raises(ValueError, match="Invalid name format"):
        invalid_upper.validate()

    # Invalid name (consecutive hyphens)
    invalid_hyphens = SkillMetadata(
        name="test--skill",  # Double hyphen
        description="Test",
        skill_path=Path("/tmp/test--skill"),
        source_type="local",
    )
    with pytest.raises(ValueError, match="consecutive hyphens"):
        invalid_hyphens.validate()


def _make_skill_zip(skill_name="test-skill", description="A test skill from ClawHub"):
    """Helper: build an in-memory ZIP containing a valid SKILL.md"""
    skill_md = (
        "---\n"
        f"name: {skill_name}\n"
        f"description: {description}\n"
        "license: MIT\n"
        "---\n"
        "# Test Skill\n\n"
        "This skill was installed from ClawHub.\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SKILL.md", skill_md)
    buf.seek(0)
    return buf.read()


class _FakeResponse:
    """Minimal httpx.Response stand-in"""

    def __init__(self, status_code=200, json_data=None, content=b"", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(f"{self.status_code}", request=None, response=self)  # type: ignore[arg-type]


def test_load_skill_from_clawhub(tmp_path):
    """Test loading a skill from ClawHub registry (latest version)"""
    loader = SkillsLoader()
    loader.cache_dir = tmp_path

    metadata_json = {
        "skill": {"slug": "test-skill"},
        "latestVersion": {"version": "1.0.0"},
    }
    zip_bytes = _make_skill_zip()

    def fake_get(url, **kwargs):
        if "/api/v1/skills/" in url:
            return _FakeResponse(json_data=metadata_json)
        if "/api/v1/download" in url:
            return _FakeResponse(content=zip_bytes)
        raise AssertionError(f"Unexpected URL: {url}")

    with patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get):
        content = loader.load_skill_from_clawhub("test-skill")

    assert content.metadata.name == "test-skill"
    assert content.metadata.source_type == "clawhub"
    assert "ClawHub" in content.body


def test_load_skill_from_clawhub_specific_version(tmp_path):
    """Test that a specific version is passed through to the API"""
    loader = SkillsLoader()
    loader.cache_dir = tmp_path

    metadata_json = {
        "skill": {"slug": "test-skill"},
        "latestVersion": {"version": "2.3.1"},
    }
    zip_bytes = _make_skill_zip()
    captured_calls: list[tuple[str, dict]] = []

    def fake_get(url, **kwargs):
        captured_calls.append((url, kwargs))
        if "/api/v1/skills/" in url:
            return _FakeResponse(json_data=metadata_json)
        if "/api/v1/download" in url:
            return _FakeResponse(content=zip_bytes)
        raise AssertionError(f"Unexpected URL: {url}")

    with patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get):
        content = loader.load_skill_from_clawhub("test-skill", version="2.3.1")

    assert content.metadata.name == "test-skill"
    download_call = [(u, kw) for u, kw in captured_calls if "/api/v1/download" in u][0]
    assert download_call[1]["params"]["version"] == "2.3.1"


def test_load_skill_from_clawhub_v_prefix_stripped(tmp_path):
    """Test that a 'v' prefix on version is stripped (v1.0.0 -> 1.0.0)"""
    loader = SkillsLoader()
    loader.cache_dir = tmp_path

    metadata_json = {
        "skill": {"slug": "test-skill"},
        "latestVersion": {"version": "1.0.0"},
    }
    zip_bytes = _make_skill_zip()
    captured_calls: list[tuple[str, dict]] = []

    def fake_get(url, **kwargs):
        captured_calls.append((url, kwargs))
        if "/api/v1/skills/" in url:
            return _FakeResponse(json_data=metadata_json)
        if "/api/v1/download" in url:
            return _FakeResponse(content=zip_bytes)
        raise AssertionError(f"Unexpected URL: {url}")

    with patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get):
        loader.load_skill_from_clawhub("test-skill", version="v1.0.0")

    download_call = [(u, kw) for u, kw in captured_calls if "/api/v1/download" in u][0]
    assert download_call[1]["params"]["version"] == "1.0.0"


def test_load_skill_from_clawhub_not_found(tmp_path):
    """Test that a 404 raises ValueError"""
    loader = SkillsLoader()
    loader.cache_dir = tmp_path

    def fake_get(url, **kwargs):
        return _FakeResponse(status_code=404)

    with (
        patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get),
        pytest.raises(ValueError, match="not found"),
    ):
        loader.load_skill_from_clawhub("nonexistent-skill")


def test_load_skill_from_clawhub_network_error(tmp_path):
    """Test that a connection error raises ValueError"""
    import httpx

    loader = SkillsLoader()
    loader.cache_dir = tmp_path

    def fake_get(url, **kwargs):
        raise httpx.ConnectError("Connection refused")

    with (
        patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get),
        pytest.raises(ValueError, match="connect"),
    ):
        loader.load_skill_from_clawhub("some-skill")


def test_load_skill_from_clawhub_rate_limited(tmp_path):
    """Test that a 429 raises ValueError with rate limit message"""
    loader = SkillsLoader()
    loader.cache_dir = tmp_path

    metadata_json = {
        "skill": {"slug": "test-skill"},
        "latestVersion": {"version": "1.0.0"},
    }

    def fake_get(url, **kwargs):
        if "/api/v1/skills/" in url:
            return _FakeResponse(json_data=metadata_json)
        if "/api/v1/download" in url:
            import time

            reset_ts = str(int(time.time()) + 42)
            return _FakeResponse(status_code=429, headers={"x-ratelimit-reset": reset_ts})
        raise AssertionError(f"Unexpected URL: {url}")

    with (
        patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get),
        pytest.raises(ValueError, match="rate limit"),
    ):
        loader.load_skill_from_clawhub("test-skill")


def test_search_clawhub():
    """Test searching ClawHub registry"""
    loader = SkillsLoader()

    search_json = {
        "results": [
            {"slug": "pdf-reader", "description": "Read PDFs", "version": "1.0.0"},
            {"slug": "web-search", "description": "Search the web", "version": "0.2.0"},
        ],
        "total": 2,
    }

    def fake_get(url, **kwargs):
        return _FakeResponse(json_data=search_json)

    with patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get):
        result = loader.search_clawhub("pdf")

    assert "pdf-reader" in result
    assert "web-search" in result
    assert "1.0.0" in result


def test_search_clawhub_no_results():
    """Test search with no results returns informative message"""
    loader = SkillsLoader()

    search_json = {"results": [], "total": 0}

    def fake_get(url, **kwargs):
        return _FakeResponse(json_data=search_json)

    with patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get):
        result = loader.search_clawhub("nonexistent-skill-xyz")

    assert "No skills found" in result


def test_search_skills_sh():
    """Test searching skills.sh registry"""
    loader = SkillsLoader()

    search_json = {
        "skills": [
            {"id": "steipete/clawdis/weather", "name": "weather", "source": "steipete/clawdis", "installs": 117},
            {
                "id": "vikiboss/60s-skills/weather-query",
                "name": "weather-query",
                "source": "vikiboss/60s-skills",
                "installs": 27,
            },
        ],
    }

    def fake_get(url, **kwargs):
        return _FakeResponse(json_data=search_json)

    with patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get):
        result = loader.search_skills_sh("weather")

    assert "weather" in result
    assert "steipete/clawdis" in result
    assert "117" in result
    # Install command must use the full skill id (org/repo/skill), not just source (org/repo)
    assert "/skill/install steipete/clawdis/weather" in result
    assert "/skill/install vikiboss/60s-skills/weather-query" in result


def test_search_skills_sh_no_results():
    """Test skills.sh search with no results"""
    loader = SkillsLoader()

    search_json = {"skills": []}

    def fake_get(url, **kwargs):
        return _FakeResponse(json_data=search_json)

    with patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get):
        result = loader.search_skills_sh("nonexistent-xyz")

    assert "No skills found" in result


@pytest.mark.parametrize("version", ["1.0.0/../../../victim", "../victim", "1.0 0", "1.0;rm"])
def test_clawhub_malicious_registry_version_rejected(tmp_path, version):
    """A version from the registry must not escape the cache dir (rmtree + extract there)."""
    cache = tmp_path / "cache"
    (cache / "clawhub_test-skill_1.0.0").mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("important")
    loader = SkillsLoader()
    loader.cache_dir = cache

    def fake_get(url, **kwargs):
        if "/api/v1/skills/" in url:
            return _FakeResponse(json_data={"latestVersion": {"version": version}})
        return _FakeResponse(content=_make_skill_zip())

    with patch("ai_assist.skills_loader.httpx.get", side_effect=fake_get), pytest.raises(ValueError):
        loader.load_skill_from_clawhub("test-skill")

    assert (victim / "keep.txt").read_text() == "important"


@pytest.mark.parametrize("slug", ["../etc", "a/b", "skill name", ""])
def test_clawhub_invalid_slug_rejected_before_request(tmp_path, slug):
    loader = SkillsLoader()
    loader.cache_dir = tmp_path

    with patch("ai_assist.skills_loader.httpx.get") as get, pytest.raises(ValueError):
        loader.load_skill_from_clawhub(slug)

    get.assert_not_called()


def test_clawhub_invalid_user_version_rejected(tmp_path):
    loader = SkillsLoader()
    loader.cache_dir = tmp_path

    with patch("ai_assist.skills_loader.httpx.get") as get, pytest.raises(ValueError):
        loader.load_skill_from_clawhub("test-skill", version="1.0/../../x")

    get.assert_not_called()
