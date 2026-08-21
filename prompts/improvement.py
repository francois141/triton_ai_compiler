import json

from utils.evaluation import (
    candidate_from_evaluation,
    evaluation_summary,
    json_default,
    normalize_nested_json,
)
from utils.traces import render_annotated_ptx_report

from .initial import IMPROVEMENT_CONTEXT_SECTION_NAMES, render_prompt_sections

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


def _improvement_context(prompt_sections):
    return render_prompt_sections(
        prompt_sections,
        IMPROVEMENT_CONTEXT_SECTION_NAMES,
    )


def _annotated_evaluation_summary(evaluation):
    summary = evaluation_summary(
        evaluation,
        include_ptx=False,
        include_ncu=False,
    )
    candidate = candidate_from_evaluation(evaluation)
    annotated_ptx = render_annotated_ptx_report(candidate.ptx, evaluation.ncu_report)
    ncu_report = {
        key: value
        for key, value in evaluation.ncu_report.items()
        if key != "source_report"
    }
    return "\n\n".join(
        (
            summary,
            "\n".join(
                (
                    "## Nsight Compute Report",
                    "```json",
                    json.dumps(
                        normalize_nested_json(ncu_report),
                        indent=2,
                        default=json_default,
                    ),
                    "```",
                )
            ),
            annotated_ptx,
        )
    )


def build_improvement_prompt(prompt_sections, best_evaluation):
    research_context = prompt_sections["float16_gemm_research"]
    research_constraint = ""
    if research_context:
        research_constraint = """
For this NCU task, use the research only to implement a metric-supported Tensor
Core or tiling change. The report remains the sole basis for bottleneck
selection, impact ordering, and performance claims.
""".strip()
    return f"""

## Kernel Source, Constexpr Values, And Launch Contract

{_improvement_context(prompt_sections)}

## Generated PTX Profiled By Nsight Compute

{_annotated_evaluation_summary(best_evaluation)}

Use the structured Nsight Compute JSON and the `// NCU ...` comments attached
to the PTX above as the only Nsight Compute evidence. The source-page CSV is
intentionally excluded; do not expect, request, or infer its missing rows.

{research_constraint}

## Planning Task

Return one to three specific, ordered improvement ideas. Each idea must target
one measurable bottleneck reported above, cite the exact metric or derived
ratio that justifies it, and prescribe one concrete PTX-level change. Order the
ideas by the expected impact supported by the report. Do not invent
micro-optimizations to fill a quota: if no further distinct, report-supported
change exists, return fewer ideas. You should propose a way to optimise this, 
explicitly mention what is the issue  and explicitly tell multiple variants 
to explore if it makes sense.


""".strip()


def build_candidate_prompt(
    prompt_sections,
    best_evaluation,
    idea,
):
    return f"""{_improvement_context(prompt_sections)}

## Current Best Verified Candidate
{_annotated_evaluation_summary(best_evaluation)}

Leading `// Tried optimization:` comments record changes that were benchmarked
but did not improve the current PTX. Do not retry those changes.

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
    prompt_sections,
    best_evaluation,
    failed_evaluation,
    idea,
    failure_analysis,
    *,
    repair_index,
    max_repair_attempts,
):
    return f"""{_improvement_context(prompt_sections)}

## Current Best Verified Candidate

{_annotated_evaluation_summary(best_evaluation)}

## Original Improvement Being Tried

Name: {idea["name"]}
Rationale: {idea["rationale"]}
Instruction: {idea["instruction"]}

## Failed Candidate And Diagnostics

{_annotated_evaluation_summary(failed_evaluation)}

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
    prompt_sections,
    failed_evaluation,
    failure_analysis,
    *,
    repair_index,
    max_repair_attempts,
):
    return f"""{_improvement_context(prompt_sections)}

## Failed Initial Candidate And Diagnostics

{_annotated_evaluation_summary(failed_evaluation)}

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
    prompt_sections,
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

    return f"""{_improvement_context(prompt_sections)}

## Failed Candidate And Complete Diagnostics

{_annotated_evaluation_summary(failed_evaluation)}
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
