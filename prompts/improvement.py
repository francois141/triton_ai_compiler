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

First decide how to improve the current best kernel. Return exactly three
specific, independent improvement ideas. Each idea must change one meaningful
performance factor only, explain why it could help, and be concrete enough to
generate one PTX candidate from it. Do not include PTX in this planning answer.

Each idea must be a localized micro-change to the current best PTX, not a
rewrite. The candidate must preserve every part of the kernel that is not
directly required by the proposed improvement: tiling, micro-tile shape,
unrolling structure, register accumulators, shared-memory staging, store
pattern, predicates, and algorithm. Do not replace the kernel with generic
loops, local-memory accumulator arrays, or a different implementation strategy.

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

## Repair Task

This is repair attempt {repair_index} of {max_repair_attempts}. Repair the
failed candidate above, not the current best candidate from scratch.

Make the smallest concrete change needed to fix the compile, verification, or
runtime failure while preserving the original improvement idea. Keep the same
PTX signature, launch metadata keys, tiling strategy, micro-tile shape,
shared-memory staging, synchronization strategy, predicate/store pattern, and
manual unroll structure unless the diagnostic proves one of those exact parts
is the bug.

Call the available `triton_ptx` tool before returning. If the repaired candidate
still fails, use the new diagnostic to make one more minimal repair while
remaining within this attempt. Return only a JSON candidate that you actually
tested with the tool; prefer the fastest verified repair. If no repair passes,
return the closest tested repair so the outer loop can record its diagnostics.
""".strip()


def build_initial_repair_prompt(
    base_prompt,
    failed_evaluation,
    *,
    repair_index,
    max_repair_attempts,
):
    return f"""{base_prompt}

## Failed Initial Candidate And Diagnostics

{evaluation_summary(failed_evaluation, include_ptx=True)}

## Initial Candidate Repair Task

This is initial candidate repair attempt {repair_index} of
{max_repair_attempts}. Repair the failed candidate above rather than generating
a fresh implementation from scratch.

Make the smallest concrete change needed to fix the compile, verification, or
runtime failure. Keep the same PTX signature, launch metadata keys, tiling
strategy, micro-tile shape, shared-memory staging, synchronization strategy,
predicate/store pattern, and manual unroll structure unless the diagnostic
proves one of those exact parts is the bug.

Call the available `triton_ptx` tool before returning. If the repaired candidate
still fails, use the new diagnostic to make one more minimal repair while
remaining within this attempt. Return only a JSON candidate that you actually
tested with the tool. If no repair passes, return the closest tested repair so
the outer loop can record its diagnostics.
""".strip()
