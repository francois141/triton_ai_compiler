"""Upload the repository's PTX documentation as an OpenAI skill."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from openai import OpenAI

PTX_SKILL_DIR = Path(__file__).resolve().parent / "ptx_skills" / "ptx_skill"


def _validate_skill_dir(skill_dir: Path) -> None:
    """Validate that a directory is an OpenAI skill package.

    Args:
        skill_dir: Directory containing the skill package.

    Raises:
        FileNotFoundError: If the directory or its ``SKILL.md`` is missing.
    """
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")
    if not (skill_dir / "SKILL.md").is_file():
        raise FileNotFoundError(f"Skill file not found: {skill_dir / 'SKILL.md'}")


def _zip_skill_dir(skill_dir: Path, zip_path: Path) -> None:
    """Create a skill ZIP preserving the package directory name.

    Args:
        skill_dir: Directory containing the skill package.
        zip_path: Destination ZIP path.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in skill_dir.rglob("*"):
            if file_path.is_file() and ".git" not in file_path.parts:
                archive.write(file_path, file_path.relative_to(skill_dir.parent))


def load_ptx(client: OpenAI | None = None) -> str:
    """Upload the bundled PTX skill and return its OpenAI skill ID.

    A caller may provide an existing ``OpenAI`` client; otherwise one is
    created from the environment. The skill package is always loaded from the
    ``skills/ptx_skills/ptx_skill`` submodule.

    Args:
        client: Optional authenticated OpenAI client.

    Returns:
        The ID assigned to the uploaded skill.

    Raises:
        FileNotFoundError: If the PTX skill submodule is not initialized.
        openai.OpenAIError: If the upload request fails.
    """
    openai_client = OpenAI() if client is None else client
    _validate_skill_dir(PTX_SKILL_DIR)

    with tempfile.TemporaryDirectory(prefix="openai_ptx_skill_") as temp_dir:
        zip_path = Path(temp_dir) / f"{PTX_SKILL_DIR.name}.zip"
        _zip_skill_dir(PTX_SKILL_DIR, zip_path)
        with zip_path.open("rb") as skill_file:
            skill = openai_client.skills.create(files=[skill_file])

    return skill.id
