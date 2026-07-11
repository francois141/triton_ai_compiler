from pathlib import Path
from shutil import rmtree

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent
TMP_DIR = ROOT_DIR / "tmp"
DATABASE_DIR = ROOT_DIR / "database"
OUTPUT_WINNER_PATTERN = "output_winner*.json"


def delete_pushed_skills(client):
    deleted_count = 0
    for skill in client.skills.list():
        print(
            "Deleting",
            skill.id,
            skill.name,
            "default=",
            skill.default_version,
            "latest=",
            skill.latest_version,
        )
        client.skills.delete(skill.id)
        deleted_count += 1
    return deleted_count


def remove_tmp_folder(tmp_dir = TMP_DIR):
    if not tmp_dir.exists():
        return False
    if not tmp_dir.is_dir():
        raise NotADirectoryError(f"Expected directory: {tmp_dir}")

    rmtree(tmp_dir)
    return True


def remove_database_folders_without_winners(
    database_dir = DATABASE_DIR,
):
    if not database_dir.exists():
        return 0
    if not database_dir.is_dir():
        raise NotADirectoryError(f"Expected directory: {database_dir}")

    removed_count = 0
    for child_dir in database_dir.iterdir():
        if not child_dir.is_dir():
            continue
        if any(child_dir.glob(OUTPUT_WINNER_PATTERN)):
            continue

        print("Removing", child_dir)
        rmtree(child_dir)
        removed_count += 1

    return removed_count


def main():
    client = OpenAI()

    deleted_skills = delete_pushed_skills(client)
    removed_tmp = remove_tmp_folder()
    removed_database_folders = remove_database_folders_without_winners()

    print(
        "Done.",
        "deleted_skills=",
        deleted_skills,
        "removed_tmp=",
        removed_tmp,
        "removed_database_folders=",
        removed_database_folders,
    )


if __name__ == "__main__":
    main()
