"""Load and manage Agent Skills following agentskills.io specification"""

import io
import logging
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config import get_config_dir
from .security import validate_tool_description

logger = logging.getLogger(__name__)

CLAWHUB_DEFAULT_REGISTRY = "https://clawhub.ai"
SKILLS_SH_DEFAULT_REGISTRY = "https://skills.sh"


class SkillMetadata(BaseModel):
    """Skill metadata from YAML frontmatter (progressive disclosure)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str  # Required, 1-64 chars
    description: str  # Required, 1-1024 chars
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    user_invocable: bool = True  # From 'user-invocable' frontmatter; False opts out of /<skill-name>
    argument_hint: str | None = None  # From 'argument-hint' frontmatter; shown in completion

    # Internal fields
    skill_path: Path
    source_type: str  # 'git' or 'local'
    source_url: str | None = None

    def validate(self):  # type: ignore[override]
        """Validate skill metadata against agentskills.io spec"""
        # Name: 1-64 chars, lowercase + hyphens, no consecutive hyphens
        if not 1 <= len(self.name) <= 64:
            raise ValueError(f"Name must be 1-64 characters: {self.name}")

        if not re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", self.name):
            raise ValueError(f"Invalid name format: {self.name}")

        if "--" in self.name:
            raise ValueError(f"Name cannot contain consecutive hyphens: {self.name}")

        # Description: 1-1024 chars
        if not 1 <= len(self.description) <= 1024:
            raise ValueError("Description must be 1-1024 characters")

        # Compatibility: max 500 chars
        if self.compatibility and len(self.compatibility) > 500:
            raise ValueError("Compatibility must be <= 500 characters")

        # Directory name must match skill name (skip for git repo root skills
        # where the directory is a cache artifact, not a skill-named directory)
        if self.source_type == "local" and self.skill_path.name != self.name:
            raise ValueError(f"Directory '{self.skill_path.name}' must match name '{self.name}'")


class SkillContent(BaseModel):
    """Full skill content including body"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: SkillMetadata
    body: str  # Markdown content after frontmatter

    # File references (loaded on-demand)
    scripts: dict[str, Path] = Field(default_factory=dict)
    references: dict[str, Path] = Field(default_factory=dict)
    assets: dict[str, Path] = Field(default_factory=dict)


class SkillsLoader:
    """Load and manage Agent Skills from multiple sources"""

    def __init__(self):
        """Initialize skills loader"""
        self.cache_dir = get_config_dir() / "skills-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_skill_from_local(self, skill_path: Path, source_type: str = "local") -> SkillContent:
        """Load a skill from a local directory

        Args:
            skill_path: Path to skill directory containing SKILL.md
            source_type: Source type for metadata ('local' or 'git')

        Returns:
            SkillContent with full skill data

        Raises:
            FileNotFoundError: If SKILL.md doesn't exist
            ValueError: If SKILL.md is invalid
        """
        skill_file = skill_path / "SKILL.md"
        if not skill_file.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_path}")

        metadata, body = self._parse_skill_file(skill_file, skill_path, source_type, None)
        return self._build_skill_content(skill_path, metadata, body)

    def _build_skill_content(self, skill_path: Path, metadata: SkillMetadata, body: str) -> SkillContent:
        """Build SkillContent from parsed metadata and body by discovering scripts, references, assets."""
        scripts = self._discover_files(skill_path / "scripts")
        references = self._discover_files(skill_path / "references")
        assets = self._discover_files(skill_path / "assets")
        return SkillContent(metadata=metadata, body=body, scripts=scripts, references=references, assets=assets)

    def load_skill_from_git(self, repo_url: str, skill_subpath: str, branch: str = "main") -> SkillContent:
        """Load a skill from a git repository

        Args:
            repo_url: Git repository URL (e.g., 'anthropics/skills')
            skill_subpath: Path to skill within repo (e.g., 'skills/pdf')
            branch: Git branch/tag to use

        Returns:
            SkillContent with full skill data

        Raises:
            FileNotFoundError: If skill doesn't exist in repo
            ValueError: If SKILL.md is invalid
        """
        # Clone/update repository
        repo_dir = self._ensure_repo_cached(repo_url, branch)

        # Navigate to skill directory
        if skill_subpath:
            skill_path = repo_dir / skill_subpath
            if not skill_path.exists():
                raise FileNotFoundError(f"Skill path '{skill_subpath}' not found in repository")
        else:
            skill_path = repo_dir

        skill_file = skill_path / "SKILL.md"
        if not skill_file.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_path}")

        metadata, body = self._parse_skill_file(skill_file, skill_path, "git", repo_url)
        return self._build_skill_content(skill_path, metadata, body)

    def load_skill_from_clawhub(self, slug: str, version: str | None = None) -> SkillContent:
        """Load a skill from the ClawHub registry.

        1. GET /api/v1/skills/{slug} — resolve latest version if none specified
        2. GET /api/v1/download?slug=...&version=... — download ZIP
        3. Extract to cache dir, load via load_skill_from_local
        """
        registry = self._get_clawhub_registry_url()
        timeout = httpx.Timeout(30.0)

        try:
            meta_resp = httpx.get(f"{registry}/api/v1/skills/{slug}", timeout=timeout)
            meta_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Skill '{slug}' not found on ClawHub registry") from e
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ValueError(f"Failed to connect to ClawHub registry: {e}") from e

        metadata = meta_resp.json()

        # Strip leading 'v' from user-provided version (e.g. v1.0.0 -> 1.0.0)
        if version and version.startswith("v"):
            version = version[1:]

        # Resolve version from metadata if not specified
        resolved_version = version or metadata.get("latestVersion", {}).get("version")

        # Build download params: only pass version if we resolved one
        download_params: dict[str, str] = {"slug": slug}
        if resolved_version:
            download_params["version"] = resolved_version
        else:
            resolved_version = "latest"

        try:
            download_resp = httpx.get(
                f"{registry}/api/v1/download",
                params=download_params,
                timeout=timeout,
                follow_redirects=True,
            )
            download_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                import time

                reset_ts = e.response.headers.get("x-ratelimit-reset", e.response.headers.get("retry-after", ""))
                try:
                    wait = max(1, int(reset_ts) - int(time.time()))
                except ValueError, TypeError:
                    wait = 60
                raise ValueError(f"ClawHub rate limit exceeded. Try again in {wait}s") from e
            raise ValueError(f"Failed to download skill '{slug}' version {resolved_version}") from e
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ValueError(f"Failed to connect to ClawHub registry: {e}") from e

        cache_name = f"clawhub_{slug}_{resolved_version}"
        skill_cache_dir = self.cache_dir / cache_name

        if skill_cache_dir.exists():
            shutil.rmtree(skill_cache_dir)
        skill_cache_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(download_resp.content)) as zf:
            # Validate all members to prevent zip slip (path traversal)
            for member in zf.namelist():
                target = (skill_cache_dir / member).resolve()
                if not target.is_relative_to(skill_cache_dir.resolve()):
                    raise ValueError(f"Zip slip detected in skill archive: {member}")
            zf.extractall(skill_cache_dir)

        return self.load_skill_from_local(skill_cache_dir, source_type="clawhub")

    def search_clawhub(self, query: str, limit: int = 10) -> str:
        """Search ClawHub registry. Returns formatted results string."""
        registry = self._get_clawhub_registry_url()

        try:
            resp = httpx.get(
                f"{registry}/api/v1/search", params={"q": query, "limit": limit}, timeout=httpx.Timeout(30.0)
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            return "Error: Failed to search ClawHub registry"
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            return f"Error: Failed to connect to ClawHub registry: {e}"

        data = resp.json()
        results = data.get("results", [])

        if not results:
            return f"No skills found matching '{query}'"

        lines = [f"ClawHub search results for '{query}' ({data.get('total', len(results))} total):\n"]
        for skill in results:
            lines.append(f"  {skill['slug']}  v{skill.get('version', '?')}")
            lines.append(f"    {skill.get('description', '')}")
            lines.append(f"    Install: /skill/install clawhub:{skill['slug']}")
            lines.append("")

        return "\n".join(lines)

    def _get_clawhub_registry_url(self) -> str:
        """Return registry URL from CLAWHUB_REGISTRY env var or default."""
        return os.environ.get("CLAWHUB_REGISTRY", CLAWHUB_DEFAULT_REGISTRY)

    def search_skills_sh(self, query: str, limit: int = 10) -> str:
        """Search skills.sh registry. Returns formatted results string."""
        registry = self._get_skills_sh_registry_url()

        try:
            resp = httpx.get(
                f"{registry}/api/search",
                params={"q": query, "limit": limit},
                timeout=httpx.Timeout(30.0),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            return "Error: Failed to search skills.sh registry"
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            return f"Error: Failed to connect to skills.sh registry: {e}"

        data = resp.json()
        results = data.get("skills", [])

        if not results:
            return f"No skills found matching '{query}'"

        lines = [f"skills.sh results for '{query}':\n"]
        for skill in results:
            skill_id = skill.get("id", "")
            source = skill.get("source", skill_id)
            installs = skill.get("installs", 0)
            installs_str = f"  ({installs} installs)" if installs else ""
            lines.append(f"  {skill['name']}{installs_str}")
            lines.append(f"    Source: {source}")
            lines.append(f"    Install: /skill/install {skill_id}")
            lines.append("")

        return "\n".join(lines)

    def _get_skills_sh_registry_url(self) -> str:
        """Return registry URL from SKILLS_SH_REGISTRY env var or default."""
        return os.environ.get("SKILLS_SH_REGISTRY", SKILLS_SH_DEFAULT_REGISTRY)

    def _parse_skill_file(
        self, skill_file: Path, skill_path: Path, source_type: str, source_url: str | None
    ) -> tuple[SkillMetadata, str]:
        """Parse SKILL.md file into metadata and body

        Args:
            skill_file: Path to SKILL.md
            skill_path: Path to skill directory
            source_type: 'git' or 'local'
            source_url: Git URL if source_type is 'git'

        Returns:
            Tuple of (SkillMetadata, body_markdown)
        """
        content = skill_file.read_text()

        # Extract YAML frontmatter
        if not content.startswith("---\n"):
            raise ValueError("SKILL.md must start with YAML frontmatter (---)")

        parts = content.split("---\n", 2)
        if len(parts) < 3:
            raise ValueError("SKILL.md must have valid YAML frontmatter delimited by ---")

        frontmatter_text = parts[1]
        body = parts[2].strip()

        # Parse YAML
        try:
            frontmatter = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML frontmatter: {e}") from e

        # Build metadata
        metadata = SkillMetadata(
            name=frontmatter["name"],
            description=frontmatter["description"],
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            metadata=frontmatter.get("metadata", {}),
            allowed_tools=frontmatter.get("allowed-tools", "").split() if frontmatter.get("allowed-tools") else [],
            user_invocable=frontmatter.get("user-invocable", True),
            argument_hint=frontmatter.get("argument-hint"),
            skill_path=skill_path,
            source_type=source_type,
            source_url=source_url,
        )

        # Validate metadata
        metadata.validate()

        # Validate skill description and body for injection patterns
        desc_warnings = validate_tool_description(f"skill:{metadata.name}", metadata.description)
        body_warnings = validate_tool_description(f"skill:{metadata.name}/body", body)
        for w in desc_warnings + body_warnings:
            logger.warning("Skill content warning for %s: %s", metadata.name, w)

        return metadata, body

    def _ensure_repo_cached(self, repo_url: str, branch: str) -> Path:
        """Clone or update git repository to cache

        Args:
            repo_url: Repository URL (e.g., 'anthropics/skills' or 'https://github.com/anthropics/skills')
            branch: Branch/tag to checkout

        Returns:
            Path to cached repository
        """
        # Normalize repo URL to full GitHub URL
        if not repo_url.startswith("http"):
            # Assume GitHub
            repo_url = f"https://github.com/{repo_url}"

        # Generate cache directory name from repo URL and branch
        # e.g., 'anthropics_skills_main'
        cache_name = repo_url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")
        cache_name = f"{cache_name}_{branch}"
        repo_cache_dir = self.cache_dir / cache_name

        if repo_cache_dir.exists():
            # Update existing repo
            try:
                subprocess.run(
                    ["git", "-C", str(repo_cache_dir), "fetch", "origin"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "-C", str(repo_cache_dir), "checkout", branch],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "-C", str(repo_cache_dir), "pull", "origin", branch],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to update repository: {e.stderr}")
                # Continue with cached version
        else:
            # Clone new repo
            try:
                subprocess.run(
                    ["git", "clone", "--branch", branch, "--depth", "1", repo_url, str(repo_cache_dir)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Failed to clone repository: {e.stderr}") from e

        return repo_cache_dir

    def _discover_files(self, directory: Path) -> dict[str, Path]:
        """Discover files in a directory

        Args:
            directory: Directory to scan

        Returns:
            Dict mapping filename to full path
        """
        if not directory.exists():
            return {}

        files = {}
        for file_path in directory.iterdir():
            if file_path.is_file():
                files[file_path.name] = file_path

        return files
