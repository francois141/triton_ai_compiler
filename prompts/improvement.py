from __future__ import annotations

import json

from utils.evaluation import evaluation_summary

from .blocks import float16_gemm_research_rules


PTX_PATCH_WORKFLOW = """
## PTX File Editing Workflow

For this task, this workflow overrides the base output contract.
The current candidate PTX is the working source file. Do not regenerate or
return the full file. To change it, call `apply_ptx_patch` with a standard
unified diff whose paths are both `candidate.ptx`; this changes only the lines
in the diff. Then call `verify_current_ptx` to compile, verify, and benchmark
that edited file. You may make further small patches and verify again. Return
only the required launch metadata after the fastest verified file is current.
""".strip()



def _float16_gemm_research_context(kernel_name):
    if kernel_name != "MatrixMultiplicationFloat16":
        return ""
    return float16_gemm_research_rules()


def build_improvement_prompt(
    _base_prompt,
    best_evaluation,
    recent_evaluations,
    kernel_name,
):
    recent_block = "\n\n".join(
        evaluation_summary(evaluation, include_ptx=False)
        for evaluation in recent_evaluations[-6:]
    )
    if not recent_block:
        recent_block = "None yet."
    research_context = _float16_gemm_research_context(kernel_name)

    return f"""## Current Best Verified Candidate

{evaluation_summary(best_evaluation, include_ptx=True)}

## Recent Candidate Outcomes

{recent_block}

{research_context}

## Planning Task

First decide how to improve the current best kernel. Return exactly three specific, 
ordered improvement ideas. Each idea must change one meaningful performance factor only, 
explain why it could help, and be concrete enough to generate one PTX candidate from it. 
Ideas two and three must be compatible with the preceding ideas: when an earlier candidate 
is verified faster, its PTX is used as the base for the next idea.
Do not include PTX in this planning answer.
""".strip()


def build_ncu_improvement_prompt(base_prompt, best_evaluation, kernel_name):
    ncu_report = best_evaluation.ncu_report
    research_context = _float16_gemm_research_context(kernel_name)
    research_constraint = ""
    if research_context:
        research_constraint = """
For this NCU task, use the research only to implement a metric-supported Tensor
Core or tiling change. The report remains the sole basis for bottleneck
selection, impact ordering, and performance claims.
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

Return exactly three specific, ordered improvement ideas. Each remaining idea must target one
measurable bottleneck reported above, cite the exact metric or derived ratio
that justifies it, and prescribe one concrete PTX-level change. Order the
remaining ideas by the expected impact supported by the report. If the report
does not support two distinct additional changes, return conservative
measurement-driven ideas that keep the kernel unchanged except for the
smallest change needed to test the cited bottleneck.

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
and benchmark every variation you choose
to investigate. You may call the tool multiple times before returning: use it
to compare technically distinct micro-variations of this same improvement and
their performance. If a variation fails compilation or correctness, repair it
using the diagnostics and call `launch_verifier` again. Return only the fastest
verified variation you actually tested. If no repair passes, return the closest
repaired candidate you tested so the outer loop can record diagnostics.
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
