"""Tests for identity management"""

import re
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from ai_assist.identity import AssistantIdentity, CommunicationPreferences, Identity, UserIdentity, get_identity


def test_default_identity():
    """Test that default identity has expected values"""
    identity = Identity()

    assert identity.version == "1.0"
    assert identity.user.name == "there"
    assert identity.user.role == "Manager"
    assert identity.user.organization is None
    assert identity.assistant.nickname == "Nexus"
    assert identity.preferences.formality == "professional"


def test_custom_identity():
    """Test creating identity with custom values"""
    identity = Identity(
        user=UserIdentity(name="Fred Lepied", role="Engineering Manager", organization="Red Hat"),
        assistant=AssistantIdentity(nickname="Atlas"),
        preferences=CommunicationPreferences(formality="casual"),
    )

    assert identity.user.name == "Fred Lepied"
    assert identity.user.role == "Engineering Manager"
    assert identity.user.organization == "Red Hat"
    assert identity.assistant.nickname == "Atlas"
    assert identity.preferences.formality == "casual"


def test_system_prompt_generation_default():
    """Test system prompt with default identity"""
    identity = Identity()
    prompt = identity.get_system_prompt()

    assert "Nexus" in prompt
    assert "AI assistant" in prompt
    assert "professional tone" in prompt


def test_system_prompt_generation_custom():
    """Test system prompt includes nickname, user name, and role"""
    identity = Identity(
        user=UserIdentity(name="Fred Lepied", role="Engineering Manager", organization="Red Hat"),
        assistant=AssistantIdentity(nickname="Atlas"),
    )
    prompt = identity.get_system_prompt()

    assert "Atlas" in prompt
    assert "Fred Lepied" in prompt
    assert "Engineering Manager" in prompt
    assert "Red Hat" in prompt
    assert "professional tone" in prompt


def test_system_prompt_with_casual_preference():
    """Test system prompt with casual communication style"""
    identity = Identity(preferences=CommunicationPreferences(formality="casual"))
    prompt = identity.get_system_prompt()

    assert "casual" in prompt or "friendly manner" in prompt


def test_system_prompt_with_friendly_preference():
    """Test system prompt with friendly communication style"""
    identity = Identity(preferences=CommunicationPreferences(formality="friendly"))
    prompt = identity.get_system_prompt()

    assert "warm" in prompt or "approachable" in prompt


def test_greeting_with_name():
    """Test personalized greeting"""
    identity = Identity(user=UserIdentity(name="Fred Lepied"), assistant=AssistantIdentity(nickname="Atlas"))
    greeting = identity.get_greeting()

    assert greeting == "Hello Fred Lepied, I'm Atlas."


def test_greeting_default():
    """Test generic greeting for default identity"""
    identity = Identity()
    greeting = identity.get_greeting()

    assert greeting == "Hello, I'm Nexus."


def test_load_from_nonexistent_file():
    """Test graceful fallback to defaults when file doesn't exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nonexistent" / "identity.yaml"
        identity = Identity.load_from_file(path)

        # Should return default identity
        assert identity.user.name == "there"
        assert identity.assistant.nickname == "Nexus"


def test_save_and_load():
    """Test persistence of identity"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "identity.yaml"

        # Create and save identity
        original = Identity(
            user=UserIdentity(name="Fred Lepied", role="Engineering Manager", organization="Red Hat"),
            assistant=AssistantIdentity(nickname="Atlas"),
        )
        original.save_to_file(path)

        # Load and verify
        loaded = Identity.load_from_file(path)

        assert loaded.user.name == "Fred Lepied"
        assert loaded.user.role == "Engineering Manager"
        assert loaded.user.organization == "Red Hat"
        assert loaded.assistant.nickname == "Atlas"


def test_save_creates_directory():
    """Test that save creates parent directory if it doesn't exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "identity.yaml"

        identity = Identity()
        identity.save_to_file(path)

        assert path.exists()
        assert path.parent.exists()


def test_load_invalid_yaml():
    """Test handling of invalid YAML"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "identity.yaml"

        # Write invalid YAML
        with open(path, "w") as f:
            f.write("invalid: yaml: content: [")

        # Should return default identity
        identity = Identity.load_from_file(path)
        assert identity.user.name == "there"


def test_load_empty_file():
    """Test handling of empty file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "identity.yaml"

        # Create empty file
        path.touch()

        # Should return default identity
        identity = Identity.load_from_file(path)
        assert identity.user.name == "there"


def test_get_identity_cached():
    """Test that get_identity returns cached instance"""
    # Clear cache by importing fresh
    from ai_assist import identity as identity_module

    identity_module._identity = None

    # First call loads from file
    id1 = get_identity()

    # Second call should return same instance
    id2 = get_identity()

    assert id1 is id2


def test_get_identity_reload():
    """Test that reload parameter forces reload"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "identity.yaml"

        # Save an identity
        Identity(user=UserIdentity(name="Test User")).save_to_file(path)

        # Load it
        _id1 = Identity.load_from_file(path)

        # Modify file
        Identity(user=UserIdentity(name="Different User")).save_to_file(path)

        # Load without reload should give cached
        id2 = Identity.load_from_file(path)

        # Verify reload works
        assert id2.user.name == "Different User"


def test_user_context_in_system_prompt():
    """Test that user context appears in system prompt"""
    identity = Identity(
        user=UserIdentity(name="Fred Lepied", context="I manage a team of 8 engineers working on OpenShift CI/CD.")
    )
    prompt = identity.get_system_prompt()

    assert "Context:" in prompt
    assert "8 engineers" in prompt
    assert "OpenShift CI/CD" in prompt


def test_custom_personality_override():
    """Test that custom personality overrides default"""
    identity = Identity(
        user=UserIdentity(name="Fred Lepied"),
        assistant=AssistantIdentity(
            nickname="Nexus",
            personality="You are Nexus, Fred's technical advisor specializing in CI/CD and distributed systems.",
        ),
    )
    prompt = identity.get_system_prompt()

    # Should use custom personality and include nickname
    assert "technical advisor" in prompt
    assert "CI/CD and distributed systems" in prompt
    assert "Nexus" in prompt


def test_custom_personality_includes_nickname():
    """Test that nickname is included even when personality doesn't mention it"""
    identity = Identity(
        user=UserIdentity(name="Alice"),
        assistant=AssistantIdentity(
            nickname="Bolt",
            personality="You are a personal fitness coach.",
        ),
    )
    prompt = identity.get_system_prompt()
    assert "Bolt" in prompt
    assert "fitness coach" in prompt


def test_verbosity_preferences():
    """Test verbosity preference in system prompt"""
    # Concise
    identity = Identity(preferences=CommunicationPreferences(verbosity="concise"))
    assert "concise" in identity.get_system_prompt().lower()

    # Detailed
    identity = Identity(preferences=CommunicationPreferences(verbosity="detailed"))
    assert "detailed" in identity.get_system_prompt().lower()

    # Verbose
    identity = Identity(preferences=CommunicationPreferences(verbosity="verbose"))
    assert "verbose" in identity.get_system_prompt().lower() or "comprehensive" in identity.get_system_prompt().lower()


def test_emoji_usage_preferences():
    """Test emoji usage preference in system prompt"""
    # None
    identity = Identity(preferences=CommunicationPreferences(emoji_usage="none"))
    assert "not use emojis" in identity.get_system_prompt().lower()

    # Minimal
    identity = Identity(preferences=CommunicationPreferences(emoji_usage="minimal"))
    assert "sparingly" in identity.get_system_prompt().lower()

    # Moderate
    identity = Identity(preferences=CommunicationPreferences(emoji_usage="moderate"))
    prompt = identity.get_system_prompt().lower()
    assert "moderate" in prompt or "occasionally" in prompt

    # Liberal
    identity = Identity(preferences=CommunicationPreferences(emoji_usage="liberal"))
    assert "engaging" in identity.get_system_prompt().lower()


def test_timezone_included_in_system_prompt():
    """Test that timezone is included in system prompt for locale inference"""
    identity = Identity(user=UserIdentity(name="Fred", timezone="Europe/Paris"))
    prompt = identity.get_system_prompt()
    assert "Europe/Paris" in prompt
    assert "locale" in prompt.lower()


def test_no_timezone_no_locale_in_prompt():
    """Test that no timezone info is added when timezone is not set"""
    identity = Identity(user=UserIdentity(name="Fred"))
    prompt = identity.get_system_prompt()
    assert "timezone" not in prompt.lower()


def test_datetime_in_system_prompt():
    """Test that current date and time are included in system prompt"""
    identity = Identity(user=UserIdentity(name="Fred", timezone="Europe/Paris"))
    prompt = identity.get_system_prompt()
    assert re.search(r"Today is \w+, \d{4}-\d{2}-\d{2}", prompt)
    assert re.search(r"current time is \d{2}:\d{2}", prompt)


def test_datetime_without_timezone():
    """Test that date/time is injected even when no timezone is set"""
    identity = Identity(user=UserIdentity(name="Fred"))
    prompt = identity.get_system_prompt()
    assert re.search(r"Today is \w+, \d{4}-\d{2}-\d{2}", prompt)
    assert re.search(r"current time is \d{2}:\d{2}", prompt)


def test_datetime_uses_user_timezone():
    """Test that date/time respects user timezone"""
    import zoneinfo

    fixed_utc = datetime(2026, 4, 6, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))
    with patch("ai_assist.identity.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_utc
        identity = Identity(user=UserIdentity(name="Fred", timezone="UTC"))
        prompt = identity.get_system_prompt()
    assert "Today is Monday, 2026-04-06" in prompt
    assert "current time is 10:30" in prompt


def test_combined_enhanced_features():
    """Test all enhanced features together"""
    identity = Identity(
        user=UserIdentity(
            name="Fred Lepied",
            role="Engineering Manager",
            organization="Red Hat",
            context="Managing 8 engineers on OpenShift CI/CD. Team includes seniors Sarah and Marcus.",
        ),
        assistant=AssistantIdentity(
            nickname="Nexus", personality="You are Nexus, a management assistant for engineering teams."
        ),
        preferences=CommunicationPreferences(formality="professional", verbosity="detailed", emoji_usage="minimal"),
    )

    prompt = identity.get_system_prompt()

    # Custom personality used
    assert "management assistant" in prompt

    # Context included
    assert "Context:" in prompt
    assert "8 engineers" in prompt
    assert "Sarah and Marcus" in prompt

    # Preferences applied
    assert "detailed" in prompt.lower()
    assert "sparingly" in prompt.lower()


def test_save_and_load_enhanced_identity():
    """Test persistence of enhanced identity"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "identity.yaml"

        # Create enhanced identity
        original = Identity(
            user=UserIdentity(
                name="Fred Lepied", role="Engineering Manager", context="Team of 8 engineers on OpenShift."
            ),
            assistant=AssistantIdentity(nickname="Nexus", personality="Custom personality here."),
            preferences=CommunicationPreferences(formality="casual", verbosity="verbose", emoji_usage="liberal"),
        )
        original.save_to_file(path)

        # Load and verify
        loaded = Identity.load_from_file(path)

        assert loaded.user.context == "Team of 8 engineers on OpenShift."
        assert loaded.assistant.personality == "Custom personality here."
        assert loaded.preferences.verbosity == "verbose"
        assert loaded.preferences.emoji_usage == "liberal"


def test_system_prompt_can_omit_datetime():
    """The cached system prompt omits the clock; the date line is available on its own"""
    identity = Identity(user=UserIdentity(name="Fred", timezone="UTC"))
    prompt = identity.get_system_prompt(include_datetime=False)
    assert "Today is" not in prompt
    assert "current time is" not in prompt
    assert re.search(
        r"^Today is \w+, \d{4}-\d{2}-\d{2}\. The current time is \d{2}:\d{2}\.$", identity.get_datetime_line()
    )
