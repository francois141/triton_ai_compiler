#!/usr/bin/env python3
import argparse
import re
from contextlib import contextmanager
from pathlib import Path

from triton_ptx.helpers.environment import is_gpu_available
from triton_ptx.helpers.triton import dump_kernel_assembly
from triton_ptx.kernels import kernel_list
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.LLMs.apertus import Apertus


DEBUG_DIRECTIVE_PATTERN = re.compile(r"^\s*\.(?:file|loc)\b")
SECTION_DIRECTIVE_PATTERN = re.compile(r"^\s*\.section\b")
FUNCTION_DIRECTIVE_PATTERN = re.compile(
    r"^\s*(?:\.(?:visible|weak|extern)\s+)*\.(?:entry|func)\b"
)
ASSEMBLY_STAGE_EXTENSIONS = {
    "ttir": ".ttir",
    "ttgir": ".ttgir",
    "llir": ".ll",
    "ptx": ".ptx",
}


@contextmanager
def disable_kernel_autotuning():
    original_init_compiled_kernels = TritonPTXKernel.init_compiled_kernels

    def init_compiled_kernels_without_autotuning(
        self,
        *,
        ptx,
        autotune=True,
        tuning_options=None,
    ):
        return original_init_compiled_kernels(
            self,
            ptx=ptx,
            autotune=False,
            tuning_options=tuning_options,
        )

    TritonPTXKernel.init_compiled_kernels = init_compiled_kernels_without_autotuning
    try:
        yield
    finally:
        TritonPTXKernel.init_compiled_kernels = original_init_compiled_kernels


def clean_ptx(ptx):
    cleaned = []
    index = 0
    in_block_comment = False
    in_string = False
    escaped = False

    while index < len(ptx):
        character = ptx[index]
        next_character = ptx[index + 1] if index + 1 < len(ptx) else ""

        if in_block_comment:
            if character == "*" and next_character == "/":
                in_block_comment = False
                index += 2
                continue
            if character == "\n":
                cleaned.append(character)
            index += 1
            continue

        if in_string:
            cleaned.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue

        if character == '"':
            in_string = True
            cleaned.append(character)
            index += 1
        elif character == "/" and next_character == "/":
            newline_index = ptx.find("\n", index)
            if newline_index == -1:
                break
            cleaned.append("\n")
            index = newline_index + 1
        elif character == "/" and next_character == "*":
            in_block_comment = True
            index += 2
        else:
            cleaned.append(character)
            index += 1

    output = []
    section_depth = 0
    expects_function_body = False

    for line in "".join(cleaned).splitlines(keepends=True):
        if section_depth:
            section_depth += line.count("{") - line.count("}")
            continue

        if SECTION_DIRECTIVE_PATTERN.match(line):
            section_depth = line.count("{") - line.count("}")
            continue

        if FUNCTION_DIRECTIVE_PATTERN.match(line):
            expects_function_body = True

        if line.lstrip().startswith("{"):
            if expects_function_body:
                expects_function_body = False
            else:
                section_depth = line.count("{") - line.count("}")
                continue

        if line.strip() and not DEBUG_DIRECTIVE_PATTERN.match(line):
            output.append(line)

    return "".join(output)


def dump_and_save_ptx_kernel(
    kernel,
    output_dir,
    clean=False,
    kernel_subdir=None,
    stages=ASSEMBLY_STAGE_EXTENSIONS,
):
    kernel_name = kernel.__class__.__name__
    assembly = dump_kernel_assembly(kernel)
    ptx = assembly.get("ptx")

    if not ptx:
        raise RuntimeError(f"Triton did not produce PTX for {kernel_name}")

    if clean:
        ptx = clean_ptx(ptx)

    output_paths = {}
    for stage, extension in stages.items():
        source = ptx if stage == "ptx" else assembly.get(stage)
        if source is None:
            continue

        stage_dir = output_dir / stage
        if kernel_subdir is not None:
            stage_dir /= kernel_subdir
        stage_dir.mkdir(parents=True, exist_ok=True)
        output_path = stage_dir / f"{kernel_name}{extension}"
        output_path.write_text(source)
        output_paths[stage] = output_path

    return output_paths


def dump_apertus_ptx(output_dir, clean, failed_kernels, kernel_name=None, stages=None):
    """Extract Apertus representations into their dedicated subdirectories."""
    stages = ASSEMBLY_STAGE_EXTENSIONS if stages is None else stages

    kernel_classes = Apertus.get_kernel_classes()
    if kernel_name is not None:
        kernel_classes = {kernel_name: kernel_classes[kernel_name]}

    for name, kernel_class in kernel_classes.items():
        kernel = kernel_class()
        kernel_name = kernel.__class__.__name__
        print(f"Compiling APERTUS/{kernel_name}")

        try:
            output_paths = dump_and_save_ptx_kernel(
                kernel,
                output_dir,
                clean,
                kernel_subdir="APERTUS",
                stages=stages,
            )
            print(f"Dumped PTX to {output_paths['ptx']}")
        except Exception as exc:
            print(f"Failed to dump APERTUS/{kernel_name}: {exc}")
            failed_kernels.append(f"APERTUS/{name}")


def main(output_dir, clean=False, apertus_kernel=None, ptx_only=False):
    if not is_gpu_available():
        raise RuntimeError("CUDA is required to compile and dump Triton PTX.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed_kernels = []
    stages = {"ptx": ".ptx"} if ptx_only else ASSEMBLY_STAGE_EXTENSIONS

    with disable_kernel_autotuning():
        if apertus_kernel is None:
            for op in kernel_list:
                kernel = op()
                kernel_name = kernel.__class__.__name__

                print(f"Compiling {kernel_name}")

                try:
                    output_paths = dump_and_save_ptx_kernel(
                        kernel,
                        output_dir,
                        clean,
                        stages=stages,
                    )
                    print(f"Dumped PTX to {output_paths['ptx']}")
                except Exception as exc:
                    print(f"Failed to dump PTX for {kernel_name}: {exc}")
                    failed_kernels.append(kernel_name)

        dump_apertus_ptx(
            output_dir,
            clean,
            failed_kernels,
            kernel_name=apertus_kernel,
            stages=stages,
        )

    if failed_kernels:
        print("\nFailed kernels:")
        for name in failed_kernels:
            print(f"- {name}")
    else:
        print("\nAll PTX files generated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile Triton operators and dump their generated PTX."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="triton_generated_ptx",
        help="Directory where generated PTX files will be saved.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove comments and .loc, .file, and .section directives from PTX.",
    )
    parser.add_argument(
        "--apertus-kernel",
        choices=sorted(Apertus.get_kernel_classes()),
        help="Extract only the specified Apertus kernel.",
    )
    parser.add_argument(
        "--ptx-only",
        action="store_true",
        help="Write PTX only, without intermediate representations.",
    )

    args = parser.parse_args()
    main(args.output_dir, args.clean, args.apertus_kernel, args.ptx_only)
