from __future__ import annotations


from utils.evaluation import evaluation_summary


KERNEL_CATEGORIES = {
    "gemm": {
        "name_markers": ("gemm", "matmul", "matrixmultiplication"),
        "planning_requirements": (
            "Use asynchronous global-to-shared data movement when the target "
            "supports it.",
            "Schedule asynchronous transfers early enough to overlap them with "
            "independent computation; do not leave them serialized.",
        ),
    },
}


def _category_requirements(kernel_name):
    normalized_name = kernel_name.lower()
    requirements = []
    for category in KERNEL_CATEGORIES.values():
        if any(marker in normalized_name for marker in category["name_markers"]):
            requirements.extend(category["planning_requirements"])
    return requirements


def build_improvement_prompt(
    base_prompt,
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

    return f"""{base_prompt}

## Planning Override

For this response only, do not generate PTX and ignore the output contract above.
Return only the structured three-idea improvement plan requested below.

## Current Best Verified Candidate

{evaluation_summary(best_evaluation, include_ptx=True)}

## Recent Candidate Outcomes

{recent_block}

## Planning Task

First decide how to improve the current best kernel. Return exactly three specific, 
ordered improvement ideas. Each idea must change one meaningful performance factor only, 
explain why it could help, and be concrete enough to generate one PTX candidate from it. 
Ideas two and three must be compatible with the preceding ideas: when an earlier candidate 
is verified faster, its PTX is used as the base for the next idea.
Do not include PTX in this planning answer.

An idea be a localized micro-change to the current best PTX and preserve every 
part of the kernel that is not directly required by the proposed improvement: 
tiling, micro-tile shape, unrolling structure, register accumulators, shared-memory staging, 
store pattern, predicates, and algorithm. Do not replace the kernel with generic loops or 
local-memory accumulator arrays.

The improvement ideas may also explore newer hardware features supported by the target architecture, 
including asynchronous copies, ldmatrix, Tensor Cores, mma.sync, or other relevant instructions, 
when the model determines that they could improve performance. 
Such an idea may include the minimum structural changes required to use the selected hardware feature correctly.

## Applicable Kernel-Category Requirements

{_category_requirements(kernel_name)}
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

Generate exactly one PTX candidate by applying only this improvement to the
current best. Preserve correctness and the required output schema.

This must be a micro-edit of the current best PTX. Preserve its tiling strategy,
micro-tile shape, manual unroll structure, register accumulators,
shared-memory staging, synchronization strategy, predicate/store pattern,
launch shape, and PTX signature unless the single improvement explicitly
requires touching one of those items. Do not generate a simpler replacement:
no generic scalar i/j/k loops, no local-memory accumulator arrays, no shorter
basic implementation, and no clean-room rewrite. The output should be
recognizably the same optimized PTX plus the requested improvement.

If the idea is an addressing/layout tweak, such as shared-memory stride,
padding, or skew, change only the relevant shared-memory allocation and address
arithmetic. Leave the compute microkernel and stores intact. If the idea turns
out not to apply, make the smallest useful related micro-change instead.

Before returning the final JSON for this candidate, call the available
`triton_ptx` tool to compile, verify, and benchmark your attempted improvement.
If it fails compilation or correctness, repair the same attempted candidate
using the diagnostics and call `triton_ptx` again. Return only the fastest
verified version you actually tested. If no repair passes, return the closest
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

You have access to the `triton_ptx` tool: use it to investigate if the failure
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

You have access to the `triton_ptx` tool: use it to investigate if the failure
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

You have access to the `triton_ptx` tool. Use it when the provided evidence is
insufficient to determine the precise defect; otherwise, do not generate a
candidate in this analysis response.
""".strip()
