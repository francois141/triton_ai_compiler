
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from openai import OpenAI

PTX_SKILL_DIR = Path(__file__).resolve().parent / "ptx_skills" / "ptx_skill"
PTX_SKILL_NAME = "ptx"
NCU_REPORT_SKILL_DIR = (
    Path(__file__).resolve().parent / "external_skills" / "ncu-report-skill"
)
NCU_REPORT_SKILL_NAME = "ncu-report-skill"
ANTHROPIC_SKILLS_BETA = "skills-2025-10-02"


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


def _find_existing_skill(client, skill_name):
    after = None

    while True:
        page = client.skills.list(after=after, limit=100, order="desc")
        for skill in page.data:
            if skill.name == skill_name:
                return skill
        if not page.has_more:
            return None
        if page.last_id is None:
            raise RuntimeError("Skill listing reported more results without a cursor.")
        after = page.last_id


def _load_openai_skill(client, skill_dir, skill_name):
    openai_client = OpenAI() if client is None else client
    _validate_skill_dir(skill_dir)

    existing_skill = _find_existing_skill(openai_client, skill_name)
    if existing_skill is not None:
        return existing_skill.id

    with tempfile.TemporaryDirectory(prefix=f"openai_{skill_name}_") as temp_dir:
        zip_path = Path(temp_dir) / f"{skill_dir.name}.zip"
        _zip_skill_dir(skill_dir, zip_path)
        with zip_path.open("rb") as skill_file:
            skill = openai_client.skills.create(files=[skill_file])

    return skill.id


def load_ptx(client=None):
    return _load_openai_skill(client, PTX_SKILL_DIR, PTX_SKILL_NAME)


def load_ncu_report(client=None):
    return _load_openai_skill(client, NCU_REPORT_SKILL_DIR, NCU_REPORT_SKILL_NAME)


def _load_anthropic_skill(client, skill_dir, skill_name):
    from anthropic.lib import files_from_dir

    _validate_skill_dir(skill_dir)
    existing_skill = next(
        (
            skill
            for skill in client.beta.skills.list(
                source="custom",
                limit=100,
                betas=[ANTHROPIC_SKILLS_BETA],
            )
            if skill.display_title == skill_name
        ),
        None,
    )
    if existing_skill is not None:
        return existing_skill.id

    skill = client.beta.skills.create(
        files=files_from_dir(skill_dir),
        display_title=skill_name,
        betas=[ANTHROPIC_SKILLS_BETA],
    )
    return skill.id


def load_anthropic_ptx(client):
    return _load_anthropic_skill(client, PTX_SKILL_DIR, PTX_SKILL_NAME)


def load_anthropic_ncu_report(client):
    return _load_anthropic_skill(
        client,
        NCU_REPORT_SKILL_DIR,
        NCU_REPORT_SKILL_NAME,
    )
