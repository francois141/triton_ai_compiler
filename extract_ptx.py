import argparse
from contextlib import contextmanager
from pathlib import Path

from triton_ptx.helpers.environment import is_gpu_available
from triton_ptx.helpers.triton import dump_kernel_assembly
from triton_ptx.kernels import kernel_list
from triton_ptx.kernels.base import TritonPTXKernel

from clean_ptx import clean_ptx

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
    ):
        return original_init_compiled_kernels(
            self,
            ptx=ptx,
            autotune=False,
        )

    TritonPTXKernel.init_compiled_kernels = init_compiled_kernels_without_autotuning
    try:
        yield
    finally:
        TritonPTXKernel.init_compiled_kernels = original_init_compiled_kernels


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


def main(output_dir, clean=False, ptx_only=False):
    if not is_gpu_available():
        raise RuntimeError("CUDA is required to compile and dump Triton PTX.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed_kernels = []
    stages = {"ptx": ".ptx"} if ptx_only else ASSEMBLY_STAGE_EXTENSIONS

    with disable_kernel_autotuning():
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
            except (RuntimeError, ValueError) as exc:
                print(f"Failed to dump PTX for {kernel_name}: {exc}")
                failed_kernels.append(kernel_name)

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
        "--ptx-only",
        action="store_true",
        help="Write PTX only, without intermediate representations.",
    )

    args = parser.parse_args()
    main(args.output_dir, args.clean, args.ptx_only)
