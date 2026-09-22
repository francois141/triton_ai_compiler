import argparse
import re
from collections.abc import Iterable
from pathlib import Path

BLOCK_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT_PATTERN = re.compile(r"//.*$")
PREDICATE_PATTERN = re.compile(r"^@!?%[A-Za-z_][A-Za-z0-9_]*(?:<\d+>)?\s+")
LABEL_PATTERN = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$.]*:\s*")
ASSIGNMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*=")
INSTRUCTION_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_.:]*)")


def instructions_in_ptx(ptx_path: Path) -> set[str]:
    source = ptx_path.read_text(encoding="utf-8", errors="replace")
    source = BLOCK_COMMENT_PATTERN.sub("", source)
    instructions = set()

    for line in source.splitlines():
        statement = LINE_COMMENT_PATTERN.sub("", line).strip()
        if not statement or statement.startswith((".", "{", "}")):
            continue
        if ASSIGNMENT_PATTERN.match(statement):
            continue

        statement = LABEL_PATTERN.sub("", statement)
        if not statement or statement.startswith("."):
            continue
        statement = PREDICATE_PATTERN.sub("", statement)
        match = INSTRUCTION_PATTERN.match(statement)
        if match is not None:
            instructions.add(match.group(1))

    return instructions


def ptx_files(astra_dir: Path) -> Iterable[Path]:
    return sorted(path for path in astra_dir.rglob("*.ptx") if path.is_file())


def collect_instructions(astra_dir: Path) -> tuple[set[str], set[str], int, int]:
    llm_instructions: set[str] = set()
    triton_instructions: set[str] = set()
    llm_files = 0
    triton_files = 0

    for ptx_path in ptx_files(astra_dir):
        instructions = instructions_in_ptx(ptx_path)
        if ptx_path.name == "triton_generated.ptx":
            triton_instructions.update(instructions)
            triton_files += 1
        else:
            llm_instructions.update(instructions)
            llm_files += 1

    return llm_instructions, triton_instructions, llm_files, triton_files


def print_instructions(title: str, instructions: set[str]) -> None:
    print(f"{title} ({len(instructions)}):")
    for instruction in sorted(instructions):
        print(f"  {instruction}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare distinct PTX instruction mnemonics in Astra LLM candidates "
            "against triton_generated.ptx baselines."
        )
    )
    parser.add_argument(
        "--astra-dir",
        type=Path,
        default=Path("astra"),
        help="directory containing Astra run directories (default: astra)",
    )
    args = parser.parse_args()

    if not args.astra_dir.is_dir():
        parser.error(f"Astra directory does not exist: {args.astra_dir}")

    llm, triton, llm_files, triton_files = collect_instructions(args.astra_dir)
    llm_only = llm - triton
    triton_only = triton - llm

    print(f"LLM PTX files: {llm_files}")
    print(f"Triton PTX files: {triton_files}")
    print(f"Distinct LLM instructions: {len(llm)}")
    print(f"Distinct Triton instructions: {len(triton)}")
    print(f"Shared instructions: {len(llm & triton)}")
    print_instructions("Instructions used by LLM but not Triton", llm_only)
    print_instructions("Instructions used by Triton but not LLM", triton_only)


if __name__ == "__main__":
    main()
