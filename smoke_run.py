from triton_api import dump_kernel_ptx, evaluate_candidate, get_kernel_data


def main():
    kernel_id = "AddKernel"
    try:
        ptx = dump_kernel_ptx(kernel_id)
        if not ptx:
            raise RuntimeError(f"Triton did not produce PTX for {kernel_id}.")

        num_warps = get_kernel_data(kernel_id)["num_warps"]
        report = evaluate_candidate(
            kernel_id,
            {
                "ptx": ptx,
                "num_threads_x": num_warps * 32,
            },
        )
    except Exception:
        print("Failure")
        return 1

    if not report.passed:
        print("Failure")
        return 1

    print("Success")
    return 0


if __name__ == "__main__":
    main()
