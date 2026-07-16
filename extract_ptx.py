#!/usr/bin/env python3
import argparse
from pathlib import Path


from triton_api import dump_kernel_ptx, is_gpu_available, list_kernels


def dump_and_save_ptx_kernel(kernel_id, output_dir):
    ptx = dump_kernel_ptx(kernel_id)

    if not ptx:
        raise RuntimeError(f"Triton did not produce PTX for {kernel_id}")

    output_path = output_dir / f"{kernel_id}.ptx"
    output_path.write_text(ptx)

    return output_path


def main(output_dir):
    if not is_gpu_available():
        raise RuntimeError("CUDA is required to compile and dump Triton PTX.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed_kernels = []

    for kernel_id in list_kernels():
        print(f"Compiling {kernel_id}")

        try:
            output_path = dump_and_save_ptx_kernel(kernel_id, output_dir)
            print(f"Dumped PTX to {output_path}")
        except Exception as exc:
            print(f"Failed to dump PTX for {kernel_id}: {exc}")
            failed_kernels.append(kernel_id)

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

    args = parser.parse_args()
    main(args.output_dir)
