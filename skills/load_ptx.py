
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from openai import OpenAI

PTX_SKILL_DIR = Path(__file__).resolve().parent / "ptx_skills" / "ptx_skill"


def _validate_skill_dir(skill_dir):
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")
    if not (skill_dir / "SKILL.md").is_file():
        raise FileNotFoundError(f"Skill file not found: {skill_dir / 'SKILL.md'}")


def _zip_skill_dir(skill_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in skill_dir.rglob("*"):
            if file_path.is_file() and ".git" not in file_path.parts:
                archive.write(file_path, file_path.relative_to(skill_dir.parent))


def load_ptx(client = None):
    openai_client = OpenAI() if client is None else client
    _validate_skill_dir(PTX_SKILL_DIR)

    with tempfile.TemporaryDirectory(prefix="openai_ptx_skill_") as temp_dir:
        zip_path = Path(temp_dir) / f"{PTX_SKILL_DIR.name}.zip"
        _zip_skill_dir(PTX_SKILL_DIR, zip_path)
        with zip_path.open("rb") as skill_file:
            skill = openai_client.skills.create(files=[skill_file])

    return skill.id
