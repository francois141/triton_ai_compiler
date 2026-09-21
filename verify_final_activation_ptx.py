import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

GPU_TYPES = ("L40S", "H100", "B200")
PRECISIONS = ("Float8", "Float16")
KERNELS = ("ReLU", "SiLU", "SwiGLU", "RMSNorm", "RoPE", "Softmax", "GELU")
ARRAY_LENGTH = 134217728
RMS_NORM_WIDTH = 4096
ROPE_HALF_WIDTH = 64
SOFTMAX_WIDTH = 256
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
GELU_BLOCK_SIZES = {
    ("B200", "Float8"): 256,
    ("B200", "Float16"): 256,
}

@dataclass(frozen=True)
class VerificationTarget:
    kernel: str
    gpu: str
    precision: str
    ptx_path: Path | None

def normalize_gpu(filename):
    if "NVIDIA_L40S_" in filename:
        return "L40S"
    if "NVIDIA_H100_" in filename:
        return "H100"
    if "NVIDIA_B200_" in filename:
        return "B200"
    return None

def find_targets(final_ptx_dir):
    targets = {}
    for kernel in KERNELS:
        for gpu in GPU_TYPES:
            for precision in PRECISIONS:
                targets[kernel, gpu, precision] = VerificationTarget(
                    kernel, gpu, precision, None
                )

        kernel_dir = final_ptx_dir / f"{kernel}Kernel"
        for ptx_path in kernel_dir.glob("*.ptx"):
            gpu = normalize_gpu(ptx_path.name)
            precision = next(
                (value for value in PRECISIONS if f"_{value}_" in ptx_path.name),
                None,
            )
            if gpu is None or precision is None:
                continue
            key = kernel, gpu, precision
            current = targets[key]
            if current.ptx_path is None or ptx_path.name > current.ptx_path.name:
                targets[key] = VerificationTarget(kernel, gpu, precision, ptx_path)
    return targets

def verifier_command(volta_bin, target, specs_dir):
    element_width = 1 if target.precision == "Float8" else 2
    spec_name = (
        {
            "RMSNorm": "rms_norm.spec",
            "RoPE": "rope.spec",
            "Softmax": "softmax.spec",
            "GELU": "gelu.spec",
        }.get(target.kernel, f"{target.kernel.lower()}.spec")
    )
    block_size = (
        RMS_NORM_BLOCK_SIZES.get((target.gpu, target.precision), 128)
        if target.kernel == "RMSNorm"
        else ROPE_BLOCK_SIZES.get((target.gpu, target.precision), "128")
        if target.kernel == "RoPE"
        else SOFTMAX_BLOCK_SIZES.get((target.gpu, target.precision), "128")
        if target.kernel == "Softmax"
        else GELU_BLOCK_SIZES.get((target.gpu, target.precision), 128)
        if target.kernel == "GELU"
        else 128
    )
    dynamic_shared_memory = (
        int(block_size) // 8 if target.kernel == "RMSNorm" else 4
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
            "1",
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

def print_table(results):
    print("| Kernel | GPU | FP8 | FP16 |")
    print("| --- | --- | --- | --- |")
    for kernel in KERNELS:
        for gpu in GPU_TYPES:
            fp8_status = results[kernel, gpu, "Float8"]
            fp16_status = results[kernel, gpu, "Float16"]
            print(f"| {kernel} | {gpu} | {fp8_status} | {fp16_status} |")

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

    results = {}
    for target in find_targets(args.final_ptx_dir).values():
        status, output = verify_target(volta_bin, target, args.specs_dir, environment)
        results[target.kernel, target.gpu, target.precision] = status
        label = f"{target.kernel} {target.gpu} {target.precision}"
        if status == "NOT VERIFIED" and target.ptx_path is not None:
            print(f"{label}: {first_output_line(output, status)}")
        if args.verbose and status == "NOT VERIFIED" and target.ptx_path is not None:
            expression = first_symbolic_expression(
                volta_bin, target, args.specs_dir, environment
            )
            print(f"{label} symbolic: {expression}")
    print_table(results)


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
        "--verbose",
        action="store_true",
        help="Print symbolic expressions for targets that do not verify.",
    )
    main(parser.parse_args())
