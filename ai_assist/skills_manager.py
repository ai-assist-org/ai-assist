"""Manage Agent Skills installation and availability"""

import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .config import get_config_dir
from .skills_loader import SkillContent, SkillsLoader

logger = logging.getLogger(__name__)


class InstalledSkill(BaseModel):
    """Record of an installed skill"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    source: str  # Original source spec (e.g., 'anthropics/skills/skills/pdf' or '/path/to/skill')
    source_type: str  # 'git', 'local', or 'clawhub'
    branch: str
    installed_at: str  # ISO timestamp
    cache_path: str  # Path to cached skill


class SkillsManager:
    """Manage Agent Skills installation and availability"""

    def __init__(self, skills_loader: SkillsLoader):
        """Initialize skills manager

        Args:
            skills_loader: SkillsLoader instance for loading skills
        """
        self.skills_loader = skills_loader
        self.installed_skills_file = get_config_dir() / "installed-skills.json"

        self.installed_skills: list[InstalledSkill] = []
        self.loaded_skills: dict[str, SkillContent] = {}

    def load_installed_skills(self):
        """Load list of installed skills from JSON and load their content"""
        if not self.installed_skills_file.exists():
            self.installed_skills = []
            self.loaded_skills = {}
            return

        try:
            with open(self.installed_skills_file) as f:
                data = json.load(f)

            self.installed_skills = []
            self.loaded_skills = {}

            for skill_data in data.get("skills", []):
                try:
                    installed_skill = InstalledSkill(**skill_data)
                    self.installed_skills.append(installed_skill)

                    # Load skill content
                    skill_path = Path(installed_skill.cache_path)
                    if skill_path.exists():
                        content = self.skills_loader.load_skill_from_local(
                            skill_path, source_type=installed_skill.source_type
                        )
                        self.loaded_skills[installed_skill.name] = content
                    else:
                        logger.warning("Skill cache not found for '%s' at %s", installed_skill.name, skill_path)

                except Exception as e:
                    logger.warning("Failed to load installed skill: %s", e)

        except json.JSONDecodeError as e:
            logger.error("Failed to parse %s: %s", self.installed_skills_file, e)
            self.installed_skills = []
            self.loaded_skills = {}

    def install_skill(self, source_spec: str) -> str:
        """Install a skill from source specification

        Args:
            source_spec: Source specification in format:
                        - Git: 'owner/repo/path/to/skill@branch'
                        - Git (top-level): 'owner/repo@branch'
                        - Git URL: 'https://github.com/owner/repo@branch'
                        - Local: '/absolute/path/to/skill@branch'

        Returns:
            Success message or error
        """
        try:
            # Parse source spec
            source, branch = self._parse_source_spec(source_spec)

            # ClawHub registry
            if source.startswith("clawhub:"):
                slug = source[len("clawhub:") :]
                source_type = "clawhub"
                version = branch if branch != "main" else None
                content = self.skills_loader.load_skill_from_clawhub(slug, version)
                cache_path = str(content.metadata.skill_path)

                # Extract installed version from cache dir name (clawhub_{slug}_{version})
                installed_version = Path(cache_path).name.rsplit("_", 1)[-1]

                skill_name = content.metadata.name

                existing = next((s for s in self.installed_skills if s.name == skill_name), None)
                if existing:
                    return f"Error: Skill '{skill_name}' is already installed. Uninstall first to reinstall."

                installed_skill = InstalledSkill(
                    name=skill_name,
                    source=source,
                    source_type=source_type,
                    branch=installed_version,
                    installed_at=datetime.now().isoformat(),
                    cache_path=cache_path,
                )

                self.installed_skills.append(installed_skill)
                self.loaded_skills[skill_name] = content
                self._save_installed_skills()

                return f"Skill '{skill_name}' installed successfully"

            # Determine source type
            source_path = Path(source).expanduser()
            if source_path.is_absolute() and source_path.exists():
                # Local path
                source_type = "local"
                content = self.skills_loader.load_skill_from_local(source_path)
                cache_path = str(source_path)  # Use original path as cache
            else:
                # Git repository
                source_type = "git"

                # Normalize GitHub URLs to owner/repo[/path] format
                source, url_branch = self._normalize_github_url(source)
                if url_branch and branch == "main":
                    branch = url_branch

                # Parse git spec: owner/repo[/path/to/skill] -> repo_url, skill_subpath
                parts = source.split("/")
                if len(parts) < 2:
                    return f"Error: Invalid git source '{source}'. Expected format: owner/repo[/path/to/skill]"

                owner = parts[0]
                repo = parts[1]
                skill_subpath = "/".join(parts[2:]) if len(parts) > 2 else ""

                repo_url = f"{owner}/{repo}"
                content = self.skills_loader.load_skill_from_git(repo_url, skill_subpath, branch)

                # Cache path is the git repo cache
                cache_path = str(content.metadata.skill_path)

            skill_name = content.metadata.name

            # Check if already installed
            existing = next((s for s in self.installed_skills if s.name == skill_name), None)
            if existing:
                return f"Error: Skill '{skill_name}' is already installed. Uninstall first to reinstall."

            # Add to installed skills
            installed_skill = InstalledSkill(
                name=skill_name,
                source=source,
                source_type=source_type,
                branch=branch,
                installed_at=datetime.now().isoformat(),
                cache_path=cache_path,
            )

            self.installed_skills.append(installed_skill)
            self.loaded_skills[skill_name] = content

            # Save to JSON
            self._save_installed_skills()

            return f"Skill '{skill_name}' installed successfully"

        except FileNotFoundError as e:
            return f"Error: {e}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: Failed to install skill: {e}"

    def uninstall_skill(self, skill_name: str) -> str:
        """Uninstall a skill by name

        Args:
            skill_name: Name of skill to uninstall

        Returns:
            Success message or error
        """
        # Find installed skill
        existing = next((s for s in self.installed_skills if s.name == skill_name), None)
        if not existing:
            return f"Error: Skill '{skill_name}' is not installed"

        # Remove from lists
        self.installed_skills.remove(existing)
        if skill_name in self.loaded_skills:
            del self.loaded_skills[skill_name]

        # Save to JSON
        self._save_installed_skills()

        return f"Skill '{skill_name}' uninstalled successfully"

    def list_installed(self) -> str:
        """Return formatted list of installed skills"""
        if not self.installed_skills:
            return "No skills installed.\n\nInstall skills with: /skill/install <source>@<branch>"

        lines = ["Installed Agent Skills:\n"]

        for skill in self.installed_skills:
            content = self.loaded_skills.get(skill.name)
            if content:
                lines.append(f"  {skill.name}")
                lines.append(f"    {content.metadata.description}")
                lines.append(f"    Source: {skill.source}@{skill.branch}")
                lines.append(f"    Installed: {skill.installed_at}")
                lines.append("")

        return "\n".join(lines)

    def get_system_prompt_section(self) -> str:
        """Generate system prompt with installed skills summary (progressive disclosure)."""
        if not self.loaded_skills:
            return ""

        sections = ["# Agent Skills\n"]
        sections.append("You have access to the following specialized skills:\n")

        for skill_name, content in self.loaded_skills.items():
            sections.append(f"- **{skill_name}**: {content.metadata.description}")

        sections.append("\nFor detailed skill instructions, call introspection__get_skill_help " "with the skill name.")

        return "\n".join(sections)

    def _normalize_github_url(self, source: str) -> tuple[str, str | None]:
        """Normalize GitHub URLs to owner/repo[/path] format.

        Handles URLs like:
            https://github.com/owner/repo -> owner/repo
            https://github.com/owner/repo/tree/branch/path -> owner/repo/path (branch extracted)
            https://github.com/owner/repo/blob/branch/path -> owner/repo/path (branch extracted)

        Args:
            source: Source string, may be a full URL or already in owner/repo format

        Returns:
            Tuple of (normalized_source, extracted_branch_or_None)
        """
        extracted_branch = None

        # Strip full GitHub URL prefix
        for prefix in ("https://github.com/", "http://github.com/"):
            if source.startswith(prefix):
                source = source[len(prefix) :]
                break

        # Remove trailing slash
        source = source.rstrip("/")

        # Remove .git suffix
        if source.endswith(".git"):
            source = source[:-4]

        # Remove /blob/<branch>/ or /tree/<branch>/ from the path
        parts = source.split("/")
        if len(parts) >= 4 and parts[2] in ("blob", "tree"):
            extracted_branch = parts[3]
            remaining = parts[4:] if len(parts) > 4 else []
            source = "/".join(parts[:2] + remaining)

        return source, extracted_branch

    def _parse_source_spec(self, source_spec: str) -> tuple[str, str]:
        """Parse source specification into source and branch

        Args:
            source_spec: Format 'source@branch'

        Returns:
            Tuple of (source, branch)
        """
        if "@" not in source_spec:
            return source_spec, "main"

        parts = source_spec.rsplit("@", 1)
        return parts[0], parts[1]

    def _save_installed_skills(self):
        """Save installed skills to JSON file"""
        data = {"skills": [skill.model_dump() for skill in self.installed_skills]}

        with open(self.installed_skills_file, "w") as f:
            json.dump(data, f, indent=2)
