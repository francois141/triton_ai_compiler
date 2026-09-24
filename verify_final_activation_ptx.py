import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

GPU_TYPES = ("L40S", "H100", "B200")
PRECISIONS = ("Float8", "Float16")
KERNELS = (
    "ReLU",
    "SiLU",
    "SwiGLU",
    "RMSNorm",
    "RoPE",
    "Softmax",
    "Lion",
    "MatrixVectorMultiplication",
    "MatrixMultiplication",
    "Sigmoid",
    "Convolution2D",
    "FusedGEMMAddSiLU",
    "BitDeltaMatrixMultiplication",
    "BitDeltaBatchedMatrixMultiplication",
)
ARRAY_LENGTH = 134217728
RMS_NORM_WIDTH = 4096
ROPE_HALF_WIDTH = 64
SOFTMAX_WIDTH = 256
LION_BLOCK_SIZE = 256
LION_ELEMENT_WIDTH = 4
MATRIX_VECTOR_MULTIPLICATION_COLS = 4096
MATRIX_VECTOR_MULTIPLICATION_FALLBACK_ROWS = 32
# The reference problem is a square 4096x4096x4096 GEMM, so this doubles as
# the full extent of every axis (M, N, and the K reduction).
MATRIX_MULTIPLICATION_DIM_FULL = 4096
MATRIX_MULTIPLICATION_FALLBACK_M = 32
MATRIX_MULTIPLICATION_FALLBACK_N = 32
MATRIX_MULTIPLICATION_SAMPLE = 256
# BitDelta packs 32 binary weights into every int32 weight element, so the
# weight buffer of the same 4096-cubed problem holds 32 times fewer elements.
BITDELTA_BITS_PER_WORD = 32
BITDELTA_PACKED_ELEMENT_WIDTH = 4
BITDELTA_BATCH_SIZE = 8
# tl.dot's cp.async-pipelined shared-memory tiles need more than the generic
# 4-byte default; sized with headroom above the largest observed
# block_k=1024 configuration's usage.
TILE_DOT_DYN_SHARED = 200000
CONVOLUTION_2D_BATCH = 32
CONVOLUTION_2D_INPUT_CHANNELS = 64
CONVOLUTION_2D_INPUT_HEIGHT = 56
CONVOLUTION_2D_INPUT_WIDTH = 56
CONVOLUTION_2D_OUTPUT_CHANNELS = 128
CONVOLUTION_2D_KERNEL_HEIGHT = 3
CONVOLUTION_2D_KERNEL_WIDTH = 3
CONVOLUTION_2D_OUTPUT_HEIGHT = (
    CONVOLUTION_2D_INPUT_HEIGHT - CONVOLUTION_2D_KERNEL_HEIGHT + 1
)
CONVOLUTION_2D_OUTPUT_WIDTH = (
    CONVOLUTION_2D_INPUT_WIDTH - CONVOLUTION_2D_KERNEL_WIDTH + 1
)
CONVOLUTION_2D_FALLBACK_BLOCK_M = 128
CONVOLUTION_2D_FALLBACK_BLOCK_N = 128
CONVOLUTION_2D_SAMPLE = 256
KERNEL_DIRS = {
    "Lion": "LionNeurIPS2023Optimizer",
    "MatrixMultiplication": "MatrixMultiplication",
    "BitDeltaMatrixMultiplication": "BitDeltaNeurIPS2024Matmul",
    "BitDeltaBatchedMatrixMultiplication": "BitDeltaNeurIPS2024BatchedMatmul",
}
RMS_NORM_BLOCK_SIZES = {
    ("L40S", "Float8"): 512,
    ("L40S", "Float16"): 256,
    ("H100", "Float8"): 256,
    ("H100", "Float16"): 512,
    ("B200", "Float8"): 512,
    ("B200", "Float16"): 512,
}
ROPE_BLOCK_SIZES = {
    ("L40S", "Float8"): "32,1",
    ("L40S", "Float16"): "128",
    ("H100", "Float8"): "32",
    ("H100", "Float16"): "128",
    ("B200", "Float8"): "128",
    ("B200", "Float16"): "128",
}
SOFTMAX_BLOCK_SIZES = {
    ("B200", "Float8"): "32,4",
}
MULTI_DIM_BLOCK_KERNELS = ("RoPE", "Softmax")

@dataclass(frozen=True)
class VerificationTarget:
    kernel: str
    gpu: str
    precision: str
    ptx_path: Path | None
    hyperparameters: dict | None = None

def normalize_gpu(filename):
    if "NVIDIA_L40S_" in filename:
        return "L40S"
    if "NVIDIA_H100_" in filename:
        return "H100"
    if "NVIDIA_B200_" in filename:
        return "B200"
    return None

def normalize_precision(filename):
    precision = next(
        (value for value in PRECISIONS if f"_{value}_" in filename), None
    )
    if precision is None and "_Unknown_" in filename:
        return "Float16"
    return precision

def load_hyperparameters(ptx_path):
    sidecar_path = ptx_path.with_suffix(".json")
    if not sidecar_path.is_file():
        return None
    try:
        hyperparameters = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return hyperparameters if isinstance(hyperparameters, dict) else None


def block_dim_from_hyperparameters(hyperparameters, key):
    if not hyperparameters:
        return None
    value = hyperparameters.get(key)
    return value if isinstance(value, int) else None


def block_size_from_hyperparameters(hyperparameters, multi_dim):
    threads_x = hyperparameters.get("num_threads_x")
    if threads_x is None:
        return None
    threads_y = hyperparameters.get("num_threads_y", 1)
    if multi_dim and threads_y not in (None, 1):
        return f"{threads_x},{threads_y}"
    return str(threads_x)


def find_targets(final_ptx_dir):
    targets = {}
    for kernel in KERNELS:
        for gpu in GPU_TYPES:
            for precision in PRECISIONS:
                targets[kernel, gpu, precision] = VerificationTarget(
                    kernel, gpu, precision, None
                )

        kernel_dir = final_ptx_dir / KERNEL_DIRS.get(kernel, f"{kernel}Kernel")
        for ptx_path in kernel_dir.glob("*.ptx"):
            gpu = normalize_gpu(ptx_path.name)
            precision = normalize_precision(ptx_path.name)
            if gpu is None or precision is None:
                continue
            key = kernel, gpu, precision
            current = targets[key]
            if current.ptx_path is None or ptx_path.name > current.ptx_path.name:
                targets[key] = VerificationTarget(
                    kernel, gpu, precision, ptx_path, load_hyperparameters(ptx_path)
                )
    return targets

# Arrays of the 4096-cubed GEMM-shaped kernels, in parameter order: name,
# element width in bytes (None follows the run's precision), whether the
# array holds BitDelta's packed weights, and the access kind.
GEMM_ARRAY_LAYOUTS = {
    "MatrixMultiplication": (
        ("a", None, False, "in"),
        ("b", None, False, "in"),
        ("c", 2, False, "out"),
    ),
    "FusedGEMMAddSiLU": (
        ("a", None, False, "in"),
        ("b", None, False, "in"),
        ("d", None, False, "in"),
        ("c", 2, False, "out"),
    ),
    "BitDeltaMatrixMultiplication": (
        ("a", None, False, "in"),
        ("b", BITDELTA_PACKED_ELEMENT_WIDTH, True, "in"),
        ("c", 2, False, "out"),
    ),
    "BitDeltaBatchedMatrixMultiplication": (
        ("a", None, False, "in"),
        ("b", BITDELTA_PACKED_ELEMENT_WIDTH, True, "in"),
        ("c", 2, False, "out"),
    ),
}
GEMM_ARRAY_BASE_STRIDE = 0x100000000
# Batch extent of the GEMM-shaped kernels whose arrays carry a leading batch
# axis; the others run a single 4096-cubed problem.
GEMM_BATCH_SIZES = {"BitDeltaBatchedMatrixMultiplication": BITDELTA_BATCH_SIZE}
# Output tile of the candidates whose sidecar does not record one. BitDelta
# decodes a grouped 64 by 128 tile, its batched form a 128 by 256 tile.
GEMM_FALLBACK_TILES = {
    "BitDeltaMatrixMultiplication": (64, 128),
    "BitDeltaBatchedMatrixMultiplication": (128, 256),
}


def gemm_arguments(arrays, batch, sample):
    arguments = []
    for index, (name, width, length, kind) in enumerate(arrays):
        base = (index + 1) * GEMM_ARRAY_BASE_STRIDE
        arguments.extend(["--array", f"{name}:{base:#x}:{width}:{length}:{kind}"])
    arguments.extend(
        argument for name, *_ in arrays for argument in ("--param", f"ptr:{name}")
    )
    arguments.extend(["--param", "int:0", "--param", "int:0"])
    axes = [("BATCH", batch)] if batch > 1 else []
    axes.extend((axis, MATRIX_MULTIPLICATION_DIM_FULL) for axis in ("M", "N", "K"))
    arguments.extend(
        argument for axis, extent in axes for argument in ("--dim", f"{axis}={extent}")
    )
    arguments.extend(["--sample", str(sample), "--verify-numeric"])
    return arguments


def gemm_grid_size(kernel, rows, cols):
    tiles = (MATRIX_MULTIPLICATION_DIM_FULL // rows) * (
        MATRIX_MULTIPLICATION_DIM_FULL // cols
    )
    # BitDelta groups both tile axes into a flat grid and takes its batch,
    # when it has one, from the second grid axis.
    if kernel == "BitDeltaMatrixMultiplication":
        return str(tiles)
    if kernel == "BitDeltaBatchedMatrixMultiplication":
        return f"{tiles},{BITDELTA_BATCH_SIZE}"
    return (
        f"{MATRIX_MULTIPLICATION_DIM_FULL // rows},"
        f"{MATRIX_MULTIPLICATION_DIM_FULL // cols}"
    )


def verifier_command(volta_bin, target, specs_dir):
    element_width = 1 if target.precision == "Float8" else 2
    spec_name = (
        {
            "RMSNorm": "rms_norm.spec",
            "RoPE": "rope.spec",
            "Softmax": "softmax.spec",
            "MatrixVectorMultiplication": "matrix_vector_multiply.spec",
            "MatrixMultiplication": "gemm.spec",
            "Convolution2D": "conv2d.spec",
            "BitDeltaMatrixMultiplication": "bitdelta_matmul.spec",
            "BitDeltaBatchedMatrixMultiplication": "bitdelta_batched_matmul.spec",
        }.get(target.kernel, f"{target.kernel.lower()}.spec")
    )
    fallback_block_size = (
        RMS_NORM_BLOCK_SIZES.get((target.gpu, target.precision), 128)
        if target.kernel == "RMSNorm"
        else ROPE_BLOCK_SIZES.get((target.gpu, target.precision), "128")
        if target.kernel == "RoPE"
        else SOFTMAX_BLOCK_SIZES.get((target.gpu, target.precision), "128")
        if target.kernel == "Softmax"
        else LION_BLOCK_SIZE
        if target.kernel == "Lion"
        else 128
    )
    hyperparameter_block_size = (
        block_size_from_hyperparameters(
            target.hyperparameters, target.kernel in MULTI_DIM_BLOCK_KERNELS
        )
        if target.hyperparameters
        else None
    )
    block_size = hyperparameter_block_size or fallback_block_size
    fallback_rows, fallback_cols = GEMM_FALLBACK_TILES.get(
        target.kernel,
        (MATRIX_MULTIPLICATION_FALLBACK_M, MATRIX_MULTIPLICATION_FALLBACK_N),
    )
    matrix_multiplication_rows = (
        block_dim_from_hyperparameters(target.hyperparameters, "block_m")
        or fallback_rows
    )
    matrix_multiplication_cols = (
        block_dim_from_hyperparameters(target.hyperparameters, "block_n")
        or fallback_cols
    )
    convolution_2d_block_m = (
        block_dim_from_hyperparameters(target.hyperparameters, "block_m")
        or CONVOLUTION_2D_FALLBACK_BLOCK_M
    )
    convolution_2d_block_n = (
        block_dim_from_hyperparameters(target.hyperparameters, "block_n")
        or CONVOLUTION_2D_FALLBACK_BLOCK_N
    )
    convolution_2d_output_rows = (
        CONVOLUTION_2D_BATCH
        * CONVOLUTION_2D_OUTPUT_HEIGHT
        * CONVOLUTION_2D_OUTPUT_WIDTH
    )
    grid_size = (
        gemm_grid_size(
            target.kernel, matrix_multiplication_rows, matrix_multiplication_cols
        )
        if target.kernel in GEMM_ARRAY_LAYOUTS
        else (
            f"{(convolution_2d_output_rows + convolution_2d_block_m - 1) // convolution_2d_block_m},"
            f"{(CONVOLUTION_2D_OUTPUT_CHANNELS + convolution_2d_block_n - 1) // convolution_2d_block_n}"
        )
        if target.kernel == "Convolution2D"
        else "1"
    )
    dynamic_shared_memory = (
        int(block_size) // 8
        if target.kernel == "RMSNorm"
        else TILE_DOT_DYN_SHARED
        if target.kernel
        in (
            "MatrixVectorMultiplication",
            "MatrixMultiplication",
            "Convolution2D",
            "FusedGEMMAddSiLU",
            "BitDeltaMatrixMultiplication",
            "BitDeltaBatchedMatrixMultiplication",
        )
        else 4
    )
    command = [
        str(volta_bin),
        "verify",
        str(target.ptx_path),
        str(specs_dir / spec_name),
    ]
    command.extend(
        [
            "--no-log-file",
            "--no-profile",
            "-k",
            "kernel",
            "-b",
            str(block_size),
            "-g",
            grid_size,
            "--dyn-shared",
            str(dynamic_shared_memory),
        ]
    )
    if target.kernel == "RMSNorm":
        command.extend(
            [
                "--array",
                f"x:0x100000000:{element_width}:{RMS_NORM_WIDTH}:in",
                "--array",
                f"weight:0x200000000:{element_width}:{RMS_NORM_WIDTH}:in",
                "--array",
                f"output:0x300000000:2:{RMS_NORM_WIDTH}:out",
                "--param",
                "ptr:x",
                "--param",
                "ptr:weight",
                "--param",
                "ptr:output",
                "--param",
                "int:0",
                "--param",
                "int:0",
                "--dim",
                "B=1",
                "--dim",
                f"D={RMS_NORM_WIDTH}",
                "--sample",
                "32",
                "--verify-numeric",
            ]
        )
        return command
    if target.kernel == "RoPE":
        command.extend(
            [
                "--array",
                f"input:0x100000000:{element_width}:{2 * ROPE_HALF_WIDTH}:in",
                "--array",
                f"cos:0x200000000:{element_width}:{2 * ROPE_HALF_WIDTH}:in",
                "--array",
                f"sin:0x300000000:{element_width}:{2 * ROPE_HALF_WIDTH}:in",
                "--array",
                f"output:0x400000000:2:{2 * ROPE_HALF_WIDTH}:out",
                "--param",
                "ptr:input",
                "--param",
                "ptr:cos",
                "--param",
                "ptr:sin",
                "--param",
                "ptr:output",
                "--param",
                "int:0",
                "--param",
                "int:0",
                "--dim",
                f"H={ROPE_HALF_WIDTH}",
                "--dim",
                "P=2",
                "--sample",
                "128",
                "--verify-numeric",
            ]
        )
        return command
    if target.kernel == "Softmax":
        rows = 16 if target.gpu == "H100" else 4
        length = rows * SOFTMAX_WIDTH
        command.extend(
            [
                "--array",
                f"input:0x100000000:{element_width}:{length}:in",
                "--array",
                f"output:0x200000000:2:{length}:out",
                "--param",
                "ptr:output",
                "--param",
                "ptr:input",
                "--param",
                f"int:{SOFTMAX_WIDTH}",
                "--param",
                f"int:{SOFTMAX_WIDTH}",
                "--param",
                f"int:{rows}",
                "--param",
                f"int:{SOFTMAX_WIDTH}",
                "--param",
                "int:0",
                "--param",
                "int:0",
                "--dim",
                f"B={rows}",
                "--dim",
                f"D={SOFTMAX_WIDTH}",
                "--sample",
                f"{length}",
                "--verify-numeric",
            ]
        )
        return command
    if target.kernel == "Lion":
        command.extend(
            [
                "--array",
                f"p:0x100000000:{LION_ELEMENT_WIDTH}:{ARRAY_LENGTH}:inout",
                "--array",
                f"grad:0x200000000:{LION_ELEMENT_WIDTH}:{ARRAY_LENGTH}:in",
                "--array",
                f"exp_avg:0x300000000:{LION_ELEMENT_WIDTH}:{ARRAY_LENGTH}:inout",
                "--param",
                "ptr:p",
                "--param",
                "ptr:grad",
                "--param",
                "ptr:exp_avg",
                "--param",
                "int:0",
                "--param",
                "int:0",
                "--dim",
                f"N={ARRAY_LENGTH}",
                "--sample",
                "256",
                "--verify-numeric",
            ]
        )
        return command
    if target.kernel == "MatrixVectorMultiplication":
        rows = (
            block_dim_from_hyperparameters(target.hyperparameters, "block_m")
            or MATRIX_VECTOR_MULTIPLICATION_FALLBACK_ROWS
        )
        cols = MATRIX_VECTOR_MULTIPLICATION_COLS
        command.extend(
            [
                "--array",
                f"a:0x100000000:{element_width}:{rows * cols}:in",
                "--array",
                f"x:0x200000000:{element_width}:{cols}:in",
                "--array",
                f"y:0x300000000:2:{rows}:out",
                "--param",
                "ptr:a",
                "--param",
                "ptr:x",
                "--param",
                "ptr:y",
                "--param",
                "int:0",
                "--param",
                "int:0",
                "--dim",
                f"M={rows}",
                "--dim",
                f"K={cols}",
                "--sample",
                f"{rows}",
                "--verify-numeric",
            ]
        )
        return command
    if target.kernel in GEMM_ARRAY_LAYOUTS:
        batch = GEMM_BATCH_SIZES.get(target.kernel, 1)
        full_elements = batch * MATRIX_MULTIPLICATION_DIM_FULL**2
        packed_elements = full_elements // BITDELTA_BITS_PER_WORD
        command.extend(
            gemm_arguments(
                [
                    (
                        name,
                        element_width if width is None else width,
                        packed_elements if packed else full_elements,
                        kind,
                    )
                    for name, width, packed, kind in GEMM_ARRAY_LAYOUTS[target.kernel]
                ],
                batch,
                min(
                    matrix_multiplication_rows * matrix_multiplication_cols,
                    MATRIX_MULTIPLICATION_SAMPLE,
                ),
            )
        )
        return command
    if target.kernel == "Convolution2D":
        batch = CONVOLUTION_2D_BATCH
        input_channels = CONVOLUTION_2D_INPUT_CHANNELS
        input_height = CONVOLUTION_2D_INPUT_HEIGHT
        input_width = CONVOLUTION_2D_INPUT_WIDTH
        output_channels = CONVOLUTION_2D_OUTPUT_CHANNELS
        kernel_height = CONVOLUTION_2D_KERNEL_HEIGHT
        kernel_width = CONVOLUTION_2D_KERNEL_WIDTH
        output_height = CONVOLUTION_2D_OUTPUT_HEIGHT
        output_width = CONVOLUTION_2D_OUTPUT_WIDTH
        output_elements = batch * output_channels * output_height * output_width
        command.extend(
            [
                "--array",
                (
                    "x_ptr:0x100000000:"
                    f"{element_width}:{batch * input_channels * input_height * input_width}:in"
                ),
                "--array",
                (
                    "weight_ptr:0x200000000:"
                    f"{element_width}:{output_channels * input_channels * kernel_height * kernel_width}:in"
                ),
                "--array",
                f"output_ptr:0x300000000:2:{output_elements}:out",
                "--param",
                "ptr:x_ptr",
                "--param",
                "ptr:weight_ptr",
                "--param",
                "ptr:output_ptr",
                "--param",
                "int:0",
                "--param",
                "int:0",
                "--dim",
                f"BATCH={batch}",
                "--dim",
                f"CIN={input_channels}",
                "--dim",
                f"IH={input_height}",
                "--dim",
                f"IW={input_width}",
                "--dim",
                f"COUT={output_channels}",
                "--dim",
                f"OH={output_height}",
                "--dim",
                f"OW={output_width}",
                "--dim",
                f"KH={kernel_height}",
                "--dim",
                f"KW={kernel_width}",
                "--sample",
                str(min(output_elements, CONVOLUTION_2D_SAMPLE)),
                "--verify-numeric",
            ]
        )
        return command
    if target.kernel == "SwiGLU":
        command.extend(
            [
                "--array",
                f"gate:0x100000000:{element_width}:{ARRAY_LENGTH}:in",
                "--array",
                f"value:0x200000000:{element_width}:{ARRAY_LENGTH}:in",
                "--array",
                f"output:0x300000000:2:{ARRAY_LENGTH}:out",
                "--param",
                "ptr:gate",
                "--param",
                "ptr:value",
                "--param",
                "ptr:output",
            ]
        )
    else:
        command.extend(
            [
                "--array",
                f"x:0x100000000:{element_width}:{ARRAY_LENGTH}:in",
                "--array",
                f"output:0x200000000:2:{ARRAY_LENGTH}:out",
                "--param",
                "ptr:x",
                "--param",
                "ptr:output",
            ]
        )
    command.extend(["--param", "int:0", "--param", "int:0"])
    command.extend(
        [
            "--dim",
            f"N={ARRAY_LENGTH}",
            "--sample",
            "256",
            "--verify-numeric",
        ]
    )
    return command

def verify_target(volta_bin, target, specs_dir, environment):
    if target.ptx_path is None:
        return "CODE ABSENT", ""
    result = subprocess.run(
        verifier_command(volta_bin, target, specs_dir),
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    output = result.stdout + result.stderr
    status = (
        "VERIFIED"
        if re.search(r"^EQUIVALENT", output, re.MULTILINE)
        else "NOT VERIFIED"
    )
    return status, output

def symbolic_command(volta_bin, target, specs_dir):
    command = verifier_command(volta_bin, target, specs_dir)
    command[1] = "analyze"
    del command[3]
    filtered_command = []
    index = 0
    options_with_value = {"--dim", "--sample"}
    while index < len(command):
        option = command[index]
        if option in options_with_value:
            index += 2
            continue
        if option == "--verify-numeric":
            index += 1
            continue
        filtered_command.append(option)
        index += 1
    # filtered_command.extend(["--print-outputs", "1"])
    return filtered_command


def first_symbolic_expression(volta_bin, target, specs_dir, environment):
    result = subprocess.run(
        symbolic_command(volta_bin, target, specs_dir),
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    output = result.stdout + result.stderr
    return next(
        (line.strip() for line in output.splitlines() if "[0] = " in line),
        first_output_line(output, "No symbolic output was produced."),
    )

def first_output_line(output, fallback):
    return next((line for line in output.splitlines() if line.strip()), fallback)

def print_table(results, kernels=KERNELS, gpus=GPU_TYPES, precisions=PRECISIONS):
    header = " | ".join(("Kernel", "GPU", *precisions))
    print(f"| {header} |")
    print(f"| {' | '.join('---' for _ in range(2 + len(precisions)))} |")
    for kernel in kernels:
        for gpu in gpus:
            statuses = [results[kernel, gpu, precision] for precision in precisions]
            print(f"| {kernel} | {gpu} | {' | '.join(statuses)} |")

def build_volta(volta_dir, z3_lib_dir):
    environment = os.environ.copy()
    if z3_lib_dir.is_dir():
        environment["RUSTFLAGS"] = f"-L {z3_lib_dir}"
    subprocess.run(
        ["cargo", "build", "-p", "volta_cli", "--release"],
        check=True,
        cwd=volta_dir,
        env=environment,
    )

def main(args):
    if not args.no_build:
        build_volta(args.volta_dir, args.z3_lib_dir)

    volta_bin = args.volta_dir / "target/release/volta"
    if not volta_bin.is_file():
        raise FileNotFoundError(f"Volta binary does not exist: {volta_bin}")

    environment = os.environ.copy()
    if args.z3_lib_dir.is_dir():
        old_library_path = environment.get("DYLD_LIBRARY_PATH", "")
        environment["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            value for value in (str(args.z3_lib_dir), old_library_path) if value
        )

    targets = find_targets(args.final_ptx_dir)
    if args.kernel is not None:
        targets = {
            key: target for key, target in targets.items() if key[0] == args.kernel
        }
    if args.gpu is not None:
        targets = {
            key: target for key, target in targets.items() if key[1] == args.gpu
        }
    if args.precision is not None:
        targets = {
            key: target for key, target in targets.items() if key[2] == args.precision
        }

    results = {}
    for target in targets.values():
        status, output = verify_target(volta_bin, target, args.specs_dir, environment)
        results[target.kernel, target.gpu, target.precision] = status
        label = f"{target.kernel} {target.gpu} {target.precision}"
        if status == "VERIFIED":
            print(f"{label}: successful")
        if status == "NOT VERIFIED" and target.ptx_path is not None:
            print(f"{label}: {first_output_line(output, status)}")
        if args.verbose and status == "NOT VERIFIED" and target.ptx_path is not None:
            expression = first_symbolic_expression(
                volta_bin, target, args.specs_dir, environment
            )
            print(f"{label} symbolic: {expression}")
    print_table(
        results,
        kernels=(args.kernel,) if args.kernel is not None else KERNELS,
        gpus=(args.gpu,) if args.gpu is not None else GPU_TYPES,
        precisions=(args.precision,) if args.precision is not None else PRECISIONS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify final ReLU and SiLU PTX across supported GPUs."
    )
    parser.add_argument("--final-ptx-dir", type=Path, default=Path("final_ptx"))
    parser.add_argument("--specs-dir", type=Path, default=Path("specs"))
    parser.add_argument("--volta-dir", type=Path, default=Path("volta"))
    parser.add_argument(
        "--z3-lib-dir", type=Path, default=Path("/opt/homebrew/opt/z3/lib")
    )
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--kernel",
        choices=KERNELS,
        default=None,
        help="Verify only the specified kernel instead of all kernels.",
    )
    parser.add_argument(
        "--gpu",
        choices=GPU_TYPES,
        default=None,
        help="Verify only the specified GPU instead of all GPUs.",
    )
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default=None,
        help="Verify only the specified precision instead of all precisions.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print symbolic expressions for targets that do not verify.",
    )
    main(parser.parse_args())
