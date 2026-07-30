from triton_ptx import Payload, TritonPTXCandidateEvaluator
from triton_ptx.helpers.triton import dump_kernel_ptx
from triton_ptx.kernels import ReLUKernel

def main():
    try:
        kernel = ReLUKernel()
        ptx = dump_kernel_ptx(kernel)
        if not ptx:
            raise RuntimeError("Triton did not produce PTX for ReLUKernel.")

        report = TritonPTXCandidateEvaluator(ReLUKernel).evaluate(
            Payload.from_input(
                {
                    "ptx": ptx,
                    "threads_x": kernel.num_warps * 32,
                }
            )
        )
    except Exception as exc:
        print("Failure")
        return 1

    if not report.passed:
        print("Failure")
        return 1

    print("Success")
    return 0


if __name__ == "__main__":
    main()
