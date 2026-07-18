"""List or delete every remote skill in the current OpenAI project."""

import argparse
import logging
import sys

from openai import APIError, OpenAI, OpenAIError


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def list_skill_ids(client):
    skill_ids = []
    after = None

    while True:
        page = client.skills.list(after=after, limit=100, order="asc")
        skill_ids.extend(skill.id for skill in page.data)
        if not page.has_more:
            return skill_ids
        if page.last_id is None:
            raise RuntimeError("Skill listing reported more results without a cursor.")
        after = page.last_id


def delete_skills(skill_ids, client):
    failures = 0
    for skill_id in skill_ids:
        try:
            client.skills.delete(skill_id)
            logger.info("Deleted %s", skill_id)
        except APIError as error:
            failures += 1
            logger.error("Could not delete %s: %s", skill_id, error)
    return failures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete all remote skills in the current OpenAI project."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Perform the irreversible deletion. Without this flag, only list skills.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        client = OpenAI()
        skill_ids = list_skill_ids(client)
    except OpenAIError as error:
        logger.error("Could not list remote skills: %s", error)
        return 1

    if not skill_ids:
        logger.info("No remote skills found in the current project.")
        return 0

    logger.info("Found %d remote skill(s):", len(skill_ids))
    for skill_id in skill_ids:
        logger.info("%s", skill_id)

    if not args.yes:
        logger.info("Dry run only. Re-run with --yes to delete them all.")
        return 0

    failures = delete_skills(skill_ids, client)
    if failures:
        logger.error("%d skill deletion(s) failed.", failures)
        return 1

    logger.info("Deleted all %d remote skill(s).", len(skill_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
