from __future__ import annotations


def system_prompt():
    return """
You are an autonomous NVIDIA PTX optimization agent. Your goal is to generate 
efficient and optimised PTX code.

## Available tools

- `launch_verifier` compiles, verifies, and benchmarks a candidate supplied as
  its four direct arguments. Its result is the source of truth for compilation,
  correctness, and performance. Always call it before returning a candidate.
- The bundled `ptx` skill provides the PTX ISA 8.7 reference in the execution
  environment. Use it to check syntax, instruction constraints, memory
  semantics, and target-architecture compatibility before using unfamiliar PTX
  features or diagnosing a PTX compilation failure.
- The bundled `ncu-report-skill` contains the complete Nsight Compute
  profiling and diagnosis reference. Use it only for NCU-backed improvement
  planning, and ground every conclusion in metrics present in the report.

## Optimization loop

1. Understand the kernel's exact semantics, signature, target, PTX version,
   launch constraints, and output contract before proposing code.
2. Create a strong candidate and call `launch_verifier`. Before returning an
   answer, try multiple technically distinct candidate variants and call the
   available tools multiple times to evaluate them. Do not present an untested
   candidate as the final answer.
3. Explicitly consider and try a shared-memory implementation whenever the
   kernel has data reuse, neighborhood access, tiling opportunities, repeated
   global-memory reads, or any access pattern where staging data in shared
   memory could plausibly improve performance. Do not skip shared memory merely
   because a simple global-memory implementation is easier to write.
4. If a candidate fails to compile or verify, use the diagnostic output to
   repair it and evaluate the repair. If it is correct, use its benchmark as
   the baseline for the next experiment.
5. Keep track of the fastest verified candidate. Change one meaningful design
   choice at a time when practical, prioritize changes likely to affect the
   bottleneck, and do not repeatedly evaluate equivalent code.
6. Never sacrifice correctness for a faster measurement. A candidate is
   eligible for the final answer only if `launch_verifier` reports that it
   compiles and is correct.
7. Produce a concrete candidate even when the implementation is difficult; use
the verifier feedback to refine it instead of falling back to a simpler kernel.
""".strip()


def improvement_planning_system_prompt():
    return """
You are an NVIDIA PTX performance-analysis agent. Your sole task is to plan
one to three evidence-based improvements for an already verified PTX kernel.
Do not generate PTX, invoke candidate-verification tools, or propose changes
outside the supplied kernel and launch contract.

Use the bundled `ncu-report-skill` as the complete Nsight Compute diagnosis
reference. Ground every conclusion in metrics or derived ratios present in the
supplied report. Treat omitted metrics as zero and do not infer unavailable
measurements.

For each proposed improvement:

1. Identify one distinct measurable bottleneck.
2. Cite the exact supporting metric or derived ratio.
3. Specify one concrete PTX-level change that addresses that bottleneck.
4. State the expected effect without claiming an unmeasured speedup.

Prioritize ideas by likely impact. Do not invent micro-optimizations merely to
reach three ideas; return fewer when the report does not support another
distinct change. Preserve correctness, the PTX signature, and launch contract.
""".strip()


def initial_task():
    return """
# Triton to Fastest PTX Conversion

You are given a Triton kernel. Generate a compile-ready PTX kernels.
The kernel must be the fastest implementation you can produce for the exact
PTX version and target listed below.
    """


def constexpr_values_block(spec):
    constexpr_params = [
        param for param in spec.parameters if _is_constexpr_annotation(param.annotation)
    ]
    if not constexpr_params:
        return "\n\n".join(
            [
                "## Operator Constexpr Values",
                "None.",
            ]
        )

    lines = []
    for param in constexpr_params:
        if param.name in spec.constexpr_values:
            lines.append(f"- {param.name}: {spec.constexpr_values[param.name]!r}")
        else:
            lines.append(f"- {param.name}: unavailable from operator constexpr_values")

    return "\n\n".join(
        [
            "## Operator Constexpr Values",
            "\n".join(
                [
                    "These tl.constexpr parameters are fixed by the operator constexpr_values dict and will be passed at launch.",
                    "Use these exact values when folding constants and writing PTX indexing logic.",
                    "Omit them from the PTX signature and do not include them in the answer dictionary.",
                    *lines,
                ]
            ),
        ]
    )


def launch_configuration_block(num_warps):
    num_threads = num_warps * 32
    return "\n\n".join(
        [
            "## Launch Configuration",
            f"The autotuner selected {num_warps} warps for this kernel.",
            f"Only launch the kernel with exactly {num_threads} threads.",
        ]
    )


def _is_constexpr_annotation(annotation):
    annotation_text = str(annotation).lower()
    return (
        annotation_text == "constexpr"
        or annotation_text.endswith(".constexpr")
        or "triton.language.core.constexpr" in annotation_text
        or ("triton.language" in annotation_text and "constexpr" in annotation_text)
    )


def signature_template(
    parameters,
    *,
    version,
    target,
    address_size,
    kernel_name="kernel",
    ptx_signature=None,
):
    runtime_params = [
        param for param in parameters if not _is_constexpr_annotation(param.annotation)
    ]

    lines = []
    for index, param in enumerate(runtime_params):
        ptx_type = (
            ptx_signature[index].ptx_type if ptx_signature is not None else ".u64"
        )
        lines.append(f"    .param {ptx_type} {param.name},")

    lines.append("    .param .u64 dummy_ptr1,")
    lines.append("    .param .u64 dummy_ptr2")

    params_block = "\n".join(lines)

    return f"""
## PTX Entry Template

Use this exact entry template and fill the body with your PTX

```ptx
.version {version}
.target {target}
.address_size {address_size}

.visible .entry {kernel_name}(
{params_block}
)
{{
   // TODO: Fill this part with your own ptx
}}
```
""".strip()


def shape_information_block(shape_information):
    return "\n\n".join(
        [
            "## Pointer Shape Information",
            "Use these exact dtype and shape details for pointer arguments.",
            shape_information,
        ]
    )


def correctness_rules(num_warps):
    num_threads = num_warps * 32
    return f"""
## Correctness Rules
- Implement the same computation and control flow as the Triton kernel.
- Respect all masks and boundary conditions exactly.
- Assume pointer inputs refer to contiguous GPU global memory unless the Triton code says otherwise.
- Treat tl.constexpr values as compile-time constants supplied by the operator defaults.
- If PTX uses one thread for one element in a constexpr-sized tile, map the {num_threads} launched threads across that tile; otherwise explicitly loop the launched threads over the full constexpr tile.
- Do not add, remove, reorder, or reinterpret runtime arguments.
""".strip()


def commenting_rules():
    return """
## PTX Commenting Rules

- Document the PTX logic with concise `//` comments written in plain human language.
- Add a short comment before each logical group of PTX instructions that explains the purpose of that group.
- Use comments to explain important indexing, masking, data movement, reductions, and stores.
- Keep comments accurate and tightly coupled to the PTX they describe.
""".strip()


def performance_rules(num_warps):
    num_threads = num_warps * 32
    return f"""
## Performance Rules

Optimize for the specific Triton kernel shown below. Use only optimizations that are semantically valid for this kernel.

Launch tuning guidance:
- `num_threads_x` should be explicitly defined for this kernel.
- If you need a multi-dimensional launch shape, you may also define `num_threads_y` and `num_threads_z`.
- The product of the provided thread dimensions must be exactly {num_threads} ({num_warps} warps), treating omitted `num_threads_y` and `num_threads_z` as 1.

Hardware rule: 
- Use modern GPU features as much as possible. Shared memory, ldmatrix, tensor cores, and async global-to-shared loads should be used when useful for the target.
""".strip()


FLOAT16_GEMM_RESEARCH_SOURCES = (
    "https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/",
    "https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/efficient_gemm.html",
    "https://www.rimikawrites.com/6-step-optimization-of-gemms-in-cuda/",
    "https://qsysarch.com/posts/gemm-kernels/",
    "https://siboehm.com/articles/22/CUDA-MMM",
    (
        "https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-"
        "Matrix-Multiplication-From-Scratch-With-Tensor-Cores.htm"
    ),
    "https://hazyresearch.stanford.edu/blog/2024-05-12-tk",
    "https://maharshi.bearblog.dev/optimizing-sgemv-cuda/",
    "https://tilelang.com/deeplearning_operators/gemv.html",
)


def float16_gemm_research_rules():
    sources = "\n".join(f"- {source}" for source in FLOAT16_GEMM_RESEARCH_SOURCES)
    return f"""
## FP16 GEMM Research and Tensor Core Requirements

This is generic FP16 GEMM optimization advice. Apply it only when it is
consistent with the operator's supplied shapes, indexing, and memory layout.
It does not change the computation, imply a 4096 x 4096 problem, or permit
treating a non-contiguous operand as a contiguous matrix.

Use the available web search tool before designing or improving a candidate to
review the following sources:
{sources}

For a true FP16 GEMM, use Tensor Cores when they are valid for the supplied
target and exact operand layout. Choose block, warp, and K tiling from the
actual problem dimensions, accounting for tensor-core tile alignment, shared
memory capacity and bank conflicts, register pressure, occupancy, coalesced
global accesses, and global-to-shared pipelining.

Treat the Tensor Core epilogue as a first-class performance-critical design,
not a final afterthought. Ensure its accumulator-to-output mapping produces
coalesced, aligned global stores with full sector utilization. When the native
accumulator layout yields strided or underutilized output stores, evaluate a
shared-memory striped epilogue: pack the FP32 accumulators to FP16, scatter a
slab into shared memory, synchronize, then issue contiguous vectorized global
stores. Reuse input-stage shared memory only after its final use. Balance any
reduced global-store transactions against the extra shared-memory traffic,
bank-conflict risk, synchronization, register pressure, and occupancy impact;
also consider a smaller slab variant when it reduces those costs. Verify every
candidate with the launch verifier; retain only the fastest correct measured
implementation.

For GEMV, tensor cores are optional. Do not use them when they are not useful.
""".strip()


def convolution_2d_float16_rules():
    return """
## Convolution Memory Layout

This operator is a 2D NCHW convolution, not a contiguous pre-materialized
GEMM input. It can be viewed logically as M=32*54*54=93312, N=128, K=576,
but the logical A[m, k] values are gathered from x_ptr and are not contiguous
in memory across the im2col K dimension. Do not use x_ptr + (m * 576 + k) or
otherwise assume an im2col buffer exists.

For output coordinates (n, oc, oh, ow), input channel c, and filter
coordinates (kh, kw), use these element offsets:
- x_ptr: ((n * 64 + c) * 56 + oh + kh) * 56 + ow + kw
- weight_ptr: ((oc * 64 + c) * 3 + kh) * 3 + kw
- output_ptr: ((n * 128 + oc) * 54 + oh) * 54 + ow

When mapping flattened m, first decode n=m/2916, then oh=(m%2916)/54 and
ow=m%54. Preserve these gathers when using shared memory or Tensor Cores;
explicitly stage the required input patch rather than assuming contiguous A
tiles. Establish a correct implementation before changing the tiling.
""".strip()


def flash_attention_float16_rules():
    return """
## Flash Attention Semantics and Layout

This is causal scaled dot-product attention over contiguous float16 tensors
with shape (batch=8, heads=16, sequence=256, head_dim=64). Each program
handles one batch-head pair and a BLOCK_M=128-row query tile. The output has
the same shape as query.

For each query row i, compute only keys j where 0 <= j <= i:
- score[i, j] = dot(query[i, :], key[j, :]) / sqrt(64)
- probability[i, :] = softmax(score[i, :]) over the valid keys only
- output[i, :] = sum(probability[i, j] * value[j, :] for valid j)

Preserve the causal mask exactly: keys above the diagonal must contribute zero
probability. Do not materialize the 256 x 256 score or probability matrix.
Use numerically stable online softmax while processing key/value tiles: retain
the running row maximum, renormalize the running output accumulator when that
maximum changes, retain the running exponential sum, then divide the final
accumulator by that sum. Accumulate dot products and softmax state in float32;
the stored output remains float16.

The supplied Triton uses an unmasked pass over earlier key blocks followed by
a masked pass over the query tile's own key block. Any PTX implementation may
use a different tiling, but it must produce the same causal attention result
for every row and batch-head pair.
""".strip()


def triton_kernel_block(source, supporting_source=""):
    source_parts = [source]
    if supporting_source:
        source_parts.insert(0, supporting_source)
    return "\n\n".join(
        [
            "## Triton Kernel",
            f"```python\n{'\n\n'.join(source_parts)}\n```",
        ]
    )


def output_contract(spec):
    num_threads = spec.num_warps * 32
    return f"""
## Output Contract

Return only a Python snippet that defines one answer dictionary named `ptx_kernel`.

The output must follow this format:

{{
    "ptx": \"\"\"<valid PTX code>\"\"\",
    "num_threads_x": <required_threads_x>,
    "num_threads_y": <optional_threads_y>,
    "num_threads_z": <optional_threads_z>,
    "difficulties": [],
}}

- generate exactly one answer;
- put the PTX code directly under the top-level `"ptx"` key;
- include `"num_threads_x"` as a positive Python integer literal for every answer;
- include `"num_threads_y"` and `"num_threads_z"` only when the kernel needs a multi-dimensional launch shape;
- include `"difficulties"` as a list of at most three concise, concrete
  verifier-reported constraints encountered while producing the candidate; do
  not use it for uncertainty, a disclaimer, or a reason to return a fallback;
  use an empty list when there are no such constraints;
- the product of the included thread dimensions must equal {num_threads} ({spec.num_warps} warps), treating omitted `"num_threads_y"` and `"num_threads_z"` as 1;
- do not include tl.constexpr parameters in the dictionary; the operator defaults are used when launching the PTX kernel;
- make the PTX string valid PTX;
- include concise human-readable PTX comments that explain the logic and each logical instruction group;
- ASCII-only;
- free of markdown fences;
""".strip()
