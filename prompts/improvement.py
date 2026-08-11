from __future__ import annotations

import json
import re

from utils.evaluation import evaluation_summary

from .blocks import float16_gemm_research_rules


PTX_PATCH_WORKFLOW = """
## PTX File Editing Workflow

For this task, this workflow overrides the usual response contract.
The current candidate PTX is the working source file. Do not regenerate or
return the full file. To change it, call `apply_ptx_patch` with a standard
unified diff whose paths are both `candidate.ptx`; this changes only the lines
in the diff. Then call `verify_current_ptx` to compile, verify, and benchmark
that edited file. You may make further small patches and verify again. Return
only the required launch metadata after the fastest verified file is current.
""".strip()


_INITIAL_ONLY_SECTION_PATTERNS = (
    r"^# Triton to Fastest PTX Conversion\n.*?(?=^## |\Z)",
    r"^## Correctness Rules\n.*?(?=^## |\Z)",
    r"^## PTX Commenting Rules\n.*?(?=^## |\Z)",
    r"^## Output Contract\n.*?(?=^## |\Z)",
)


def _improvement_base_prompt(base_prompt):
    for pattern in _INITIAL_ONLY_SECTION_PATTERNS:
        base_prompt = re.sub(
            pattern,
            "",
            base_prompt,
            flags=re.MULTILINE | re.DOTALL,
        )
    return base_prompt.strip()


def _float16_gemm_research_context(kernel_name):
    if kernel_name != "MatrixMultiplicationFloat16":
        return ""
    return float16_gemm_research_rules()


def build_improvement_prompt(base_prompt, best_evaluation, kernel_name):
    base_prompt = _improvement_base_prompt(base_prompt)
    ncu_report = best_evaluation.ncu_report
    research_context = _float16_gemm_research_context(kernel_name)
    research_constraint = ""
    if research_context:
        research_constraint = """
For this NCU task, use the research only to implement a metric-supported Tensor
Core or tiling change. The report remains the sole basis for bottleneck
selection, impact ordering, and performance claims.

For `MatrixMultiplicationFloat16`, the candidate must first match the reference
CUTLASS configuration encoded by the local direct kernel exactly: row-major
FP16 A and B plus row-major FP16 output; FP32 `mma.sync` accumulation; a
128x256x32 thread-block tile; 2x4 64x64x32 warp tiles (eight warps, 256
threads); 16x8x16 `mma.sync.aligned.row.col.f32.f16.f16.f32` tiles; and an
eight-element FP16 epilogue vector. It MUST use the direct kernel's three-stage
asynchronous mainloop (`MMA_STAGES = 3`), 16-byte `cp.async.cg` loads with the
`L2::128B` hint, CUTLASS-compatible shared-memory layouts, and 72 KiB of shared
memory (36,864 FP16 elements). This is the best known direct-CUTLASS baseline:
make the candidate conform before evaluating an optimization or comparing
performance. Test deviations from this configuration only as a single,
report-supported experiment, and retain the direct configuration if no
deviation is faster.
""".strip()
    return f"""Make every bottleneck selection and impact-ordering decision exclusively
from the Nsight Compute report below. Do not infer a bottleneck from a missing
metric. Use the source and PTX context only to formulate a concrete,
implementable PTX-level change for a report-supported bottleneck.

## Kernel Source, Constexpr Values, And Launch Contract

{base_prompt}

## Generated PTX Profiled By Nsight Compute

{evaluation_summary(best_evaluation, include_ptx=True)}

Use the bundled `ncu-report-skill` as the full Nsight Compute analysis and
diagnosis reference. Follow its report-to-diagnosis workflow and consult its
`reference/05-analysis-dimensions.md` and
`reference/06-diagnosis-playbook.md` before selecting a bottleneck. Use its
B200 metric-name mapping only when the supplied report identifies compatible
hardware; the metrics below remain the sole evidence for this plan.

## Full Nsight Compute Report

The complete unmodified NCU report is included below, including availability,
return status, diagnostics, summary, and every collected metric.

{json.dumps(ncu_report, indent=2)}

{research_context}

{research_constraint}

## Planning Task

Return one to three specific, ordered improvement ideas. Each idea must target
one measurable bottleneck reported above, cite the exact metric or derived
ratio that justifies it, and prescribe one concrete PTX-level change. Order the
ideas by the expected impact supported by the report. Do not invent
micro-optimizations to fill a quota: if no further distinct, report-supported
change exists, return fewer ideas.

One candidate idea to assess is enlarging the per-thread accumulator tile to
issue 64 Tensor Core `mma.sync` calls per tile. Consider it only if the report
shows that additional Tensor Core work per shared-memory tile could help and
that the resulting register-pressure and occupancy tradeoff is acceptable.

""".strip()


def build_candidate_prompt(
    base_prompt,
    best_evaluation,
    idea,
):
    base_prompt = _improvement_base_prompt(base_prompt)
    return f"""{base_prompt}

## Current Best Verified Candidate
{evaluation_summary(best_evaluation, include_ptx=True)}


## Single Improvement To Try

Name: {idea["name"]}
Rationale: {idea["rationale"]}
Instruction: {idea["instruction"]}

Generate PTX candidate variations that apply only this improvement to the
current best. Preserve correctness and the required output schema.


{PTX_PATCH_WORKFLOW}

Before returning the final metadata, use the patch workflow to compile, verify,
and benchmark every variation requested by the improvement instruction. For a
parameter sweep (for example, tiling or unrolling values), test every valid
listed option. You may call the tool multiple times before returning. Compare
the variations and return only the fastest verified candidate you actually
tested. If a variation fails compilation or correctness, repair it using the
diagnostics and call `launch_verifier` again. If no repair passes, return the
closest repaired candidate you tested so the outer loop can record diagnostics.
""".strip()


def build_repair_prompt(
    base_prompt,
    best_evaluation,
    failed_evaluation,
    idea,
    failure_analysis,
    *,
    repair_index,
    max_repair_attempts,
):
    base_prompt = _improvement_base_prompt(base_prompt)
    return f"""{base_prompt}

## Current Best Verified Candidate

{evaluation_summary(best_evaluation, include_ptx=True)}

## Original Improvement Being Tried

Name: {idea["name"]}
Rationale: {idea["rationale"]}
Instruction: {idea["instruction"]}

## Failed Candidate And Diagnostics

{evaluation_summary(failed_evaluation, include_ptx=True)}

## Failure Analysis And Required Fix

Root cause: {failure_analysis["root_cause"]}
Required repair: {failure_analysis["repair_instruction"]}

## Repair Task

This is repair attempt {repair_index} of {max_repair_attempts}. Repair the
failed candidate above, not the current best candidate from scratch.

Make the smallest concrete change needed to fix the compile, verification, or
runtime failure described above while preserving the original improvement idea.
Keep the same PTX signature, launch metadata keys, tiling strategy, micro-tile
shape,
shared-memory staging, synchronization strategy, predicate/store pattern, and
manual unroll structure unless the diagnostic proves one of those exact parts
is the bug.


{PTX_PATCH_WORKFLOW}

Use the bundled `ptx` skill to check relevant PTX ISA constraints. Use the
patch workflow to investigate if the failure
analysis or diagnostics leave anything uncertain, and always use it to test the
repair before returning. If the repaired candidate still fails, use the new
diagnostic to make one more minimal repair while remaining within this attempt.
Return only a JSON candidate that you actually
tested with the tool; prefer the fastest verified repair. If no repair passes,
return the closest tested repair so the outer loop can record its diagnostics.
""".strip()


def build_initial_repair_prompt(
    base_prompt,
    failed_evaluation,
    failure_analysis,
    *,
    repair_index,
    max_repair_attempts,
):
    base_prompt = _improvement_base_prompt(base_prompt)
    return f"""{base_prompt}

## Failed Initial Candidate And Diagnostics

{evaluation_summary(failed_evaluation, include_ptx=True)}

## Failure Analysis And Required Fix

Root cause: {failure_analysis["root_cause"]}
Required repair: {failure_analysis["repair_instruction"]}

## Initial Candidate Repair Task

This is initial candidate repair attempt {repair_index} of
{max_repair_attempts}. Repair the failed candidate above rather than generating
a fresh implementation from scratch.

Make the smallest concrete change needed to fix the compile, verification, or
runtime failure described above. Keep the same PTX signature, launch metadata
keys, tiling strategy, micro-tile shape, shared-memory staging, synchronization
strategy,
predicate/store pattern, and manual unroll structure unless the diagnostic
proves one of those exact parts is the bug.

{PTX_PATCH_WORKFLOW}

Use the bundled `ptx` skill to check relevant PTX ISA constraints. Use the
patch workflow to investigate if the failure
analysis or diagnostics leave anything uncertain, and always use it to test the
repair before returning. If the repaired candidate still fails, use the new
diagnostic to make one more minimal repair while remaining within this attempt.
Return only a JSON candidate that you actually
tested with the tool. If no repair passes, return the closest tested repair so
the outer loop can record its diagnostics.
""".strip()


def build_failure_analysis_prompt(
    base_prompt,
    failed_evaluation,
    *,
    repair_index,
    max_repair_attempts,
    idea=None,
):
    base_prompt = _improvement_base_prompt(base_prompt)
    improvement_context = ""
    if idea is not None:
        improvement_context = f"""
## Original Improvement Being Tried

Name: {idea["name"]}
Rationale: {idea["rationale"]}
Instruction: {idea["instruction"]}
"""

    return f"""{base_prompt}

## Failed Candidate And Complete Diagnostics

{evaluation_summary(failed_evaluation, include_ptx=True)}
{improvement_context}
## Failure Analysis Task

This is analysis for repair attempt {repair_index} of {max_repair_attempts}.
Identify the specific defect in the failed PTX candidate, using the complete
sanitizer diagnostics and candidate PTX above. Return a concise root cause and
a single, concrete repair instruction. Do not propose a rewrite or a generic
debugging checklist. Preserve the candidate's tiling, micro-tile shape,
shared-memory staging, synchronization strategy, predicates, stores, and
manual unroll structure unless the diagnostics establish that one is faulty.


Use the bundled `ptx` skill to check relevant PTX ISA constraints. Use
`launch_verifier` when the provided evidence is
insufficient to determine the precise defect; otherwise, do not generate a
candidate in this analysis response.
""".strip()
