"""Identity management for ai-assist"""

import logging
import zoneinfo
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .config import get_config_dir

logger = logging.getLogger(__name__)


class UserIdentity(BaseModel):
    """User identity information"""

    name: str = "there"
    role: str = "Manager"
    organization: str | None = None
    timezone: str | None = None
    context: str | None = None  # Detailed work context (team structure, priorities, etc.)


class AssistantIdentity(BaseModel):
    """Assistant identity information"""

    nickname: str = "Nexus"
    personality: str | None = None  # Custom personality override


class CommunicationPreferences(BaseModel):
    """Communication style preferences"""

    formality: str = "professional"  # professional, casual, friendly
    verbosity: str = "concise"  # concise, detailed, verbose
    emoji_usage: str = "moderate"  # none, minimal, moderate, liberal


class Identity(BaseModel):
    """Complete identity configuration"""

    version: str = "1.0"
    user: UserIdentity = Field(default_factory=UserIdentity)
    assistant: AssistantIdentity = Field(default_factory=AssistantIdentity)
    preferences: CommunicationPreferences = Field(default_factory=CommunicationPreferences)

    @classmethod
    def load_from_file(cls, path: Path | None = None) -> Identity:
        """Load identity from YAML file

        Args:
            path: Path to identity.yaml file. If None, uses <config_dir>/identity.yaml

        Returns:
            Identity object (uses defaults if file doesn't exist)
        """
        if path is None:
            path = get_config_dir() / "identity.yaml"

        if not path.exists():
            # Return default identity
            return cls()

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            if not data:
                return cls()

            return cls(**data)

        except yaml.YAMLError, TypeError, ValueError:
            logger.warning("Error loading identity from %s; using default identity", path, exc_info=True)
            return cls()

    def save_to_file(self, path: Path | None = None):
        """Save identity to YAML file

        Args:
            path: Path to identity.yaml file. If None, uses <config_dir>/identity.yaml
        """
        if path is None:
            path = get_config_dir() / "identity.yaml"

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_system_prompt(self) -> str:
        """Generate system prompt for Claude

        Returns:
            System prompt string that introduces the assistant
        """
        # If custom personality is provided, use it as the base
        if self.assistant.personality:
            prompt = f"Your name is {self.assistant.nickname}. {self.assistant.personality}"
        else:
            # Generate default personality
            parts = [f"You are {self.assistant.nickname}, an AI assistant"]

            if self.user.name != "there":
                # Personalized introduction
                parts.append(f"helping {self.user.name}")

                if self.user.role:
                    parts[-1] += f", a {self.user.role}"

                if self.user.organization:
                    parts[-1] += f" at {self.user.organization}"

            prompt = " ".join(parts) + "."

            # Add communication style
            if self.preferences.formality == "professional":
                prompt += " You maintain a professional tone."
            elif self.preferences.formality == "casual":
                prompt += " You communicate in a casual, friendly manner."
            elif self.preferences.formality == "friendly":
                prompt += " You are warm and approachable in your communication."

        # Add user timezone/locale context
        if self.user.timezone:
            prompt += f"\n\nThe user's timezone is {self.user.timezone}. Use this to infer their locale and language preferences (e.g. for web searches, date formats)."

        # Add user context (team structure, priorities, etc.)
        if self.user.context:
            prompt += f"\n\nContext: {self.user.context}"

        # Add verbosity preference
        if self.preferences.verbosity == "concise":
            prompt += " Be concise and to the point."
        elif self.preferences.verbosity == "detailed":
            prompt += " Provide detailed explanations when appropriate."
        elif self.preferences.verbosity == "verbose":
            prompt += " Provide comprehensive, thorough explanations."

        # Add emoji usage preference
        if self.preferences.emoji_usage == "none":
            prompt += " Do not use emojis."
        elif self.preferences.emoji_usage == "minimal":
            prompt += " Use emojis sparingly, only when they add clarity."
        elif self.preferences.emoji_usage == "moderate":
            prompt += " Use emojis occasionally to enhance communication."
        elif self.preferences.emoji_usage == "liberal":
            prompt += " Feel free to use emojis to make communication more engaging."

        # Add current date and time
        now = datetime.now()
        if self.user.timezone:
            try:
                tz = zoneinfo.ZoneInfo(self.user.timezone)
                now = datetime.now(tz)
            except Exception:
                pass
        prompt += f"\n\nToday is {now.strftime('%A')}, {now.strftime('%Y-%m-%d')}. The current time is {now.strftime('%H:%M')}."

        return prompt

    def get_greeting(self) -> str:
        """Get personalized greeting

        Returns:
            Greeting string
        """
        if self.user.name != "there":
            return f"Hello {self.user.name}, I'm {self.assistant.nickname}."
        else:
            return f"Hello, I'm {self.assistant.nickname}."


# Module-level cached identity
_identity: Identity | None = None


def get_identity(reload: bool = False) -> Identity:
    """Get the current identity (cached)

    Args:
        reload: If True, reload from file instead of using cache

    Returns:
        Identity object
    """
    global _identity

    if _identity is None or reload:
        _identity = Identity.load_from_file()

    return _identity
