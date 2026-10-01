<p align="center">
  <img alt="Triton AI Compiler logo" src="logo.png" width="200">
</p>

# Triton AI Compiler

<p align="center">
  <strong>The agent client for PTX Gym: it drives an LLM through the loop of
  writing, verifying, and benchmarking PTX for a Triton kernel.</strong>
</p>

<p align="center">
  <img alt="Status: research preview" src="https://img.shields.io/badge/status-research%20preview-orange">
  <img alt="Role: PTX Gym client" src="https://img.shields.io/badge/role-PTX%20Gym%20client-blueviolet">
  <img alt="Task: Triton to PTX" src="https://img.shields.io/badge/task-Triton%20%E2%86%92%20PTX-blue">
  <img alt="Hardware: NVIDIA GPU" src="https://img.shields.io/badge/hardware-NVIDIA%20GPU-76B900">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#quick-run">Quick run</a> ·
  <a href="#agent-options">Agent options</a> ·
  <a href="#run-outputs">Run outputs</a> ·
  <a href="#utilities">Utilities</a>
</p>

## Overview

[PTX Gym](ptx_gym/README.md) is the environment: it fixes the compilation
contract for a kernel, evaluates a candidate PTX against it, and reports
correctness, formal verification, and speed. It does not decide what PTX to
try next.

**This repository is the client.** It is a tool-calling agent for OpenAI,
Anthropic, and OpenRouter models. It prompts a model for PTX, submits each
candidate to PTX Gym, reads back the verdict and the Nsight Compute report,
plans the next edit, and records every candidate it evaluated.

The two are kept separate on purpose: PTX Gym is the benchmark and stays
independent of any agent, while this client is free to change its prompting,
provider, and search strategy. PTX Gym is vendored here as the `ptx_gym`
submodule, so one checkout gives you both.

## Installation

> [!CAUTION]
> Model-generated PTX is untrusted low-level code. Run the agent on isolated,
> non-production machines.

### Install

```bash
git clone --recursive https://github.com/francois141/triton_ai_compiler
cd triton_ai_compiler

uv venv .triton_ai_compiler
source .triton_ai_compiler/bin/activate

cd ptx_gym
uv pip install torch numpy
MAX_JOBS=64 uv pip install -e . -v   # builds the patched Triton and ptx_gym
cd ..

uv pip install anthropic openai matplotlib pydantic tiktoken
```

`MAX_JOBS` caps the parallelism of the Triton build; lower it on smaller
machines.

### TODO: Add the instruction to fetch the previous results

### Environment variables

Set the API key for the provider you use. The GPU tool paths are read by PTX
Gym while it evaluates a candidate.

| Variable | Needed for |
| --- | --- |
| `OPENAI_API_KEY` | `--provider openai` (the default). |
| `ANTHROPIC_API_KEY` | `--provider anthropic`. |
| `OPENROUTER_API_KEY` | `--provider openrouter`. |
| `OPENAI_BASE_URL` | Optional. Points the OpenAI provider at an OpenAI-compatible endpoint. |
| `PTX_MEMORY_SANITIZER` | Path to `compute-sanitizer`. Required unless you pass `--disable-sanitizer`. |
| `NCU_PATH` | Path to `ncu`. Without it, planning falls back to source-only hypotheses. |

```bash
export OPENAI_API_KEY=sk-...
export PTX_MEMORY_SANITIZER=$(which compute-sanitizer)
export NCU_PATH=$(which ncu)
```

## Quick start

Check that the GPU, the patched Triton, and the kernel suite work, without
calling any model:

```bash
python run_kernel.py SoftmaxFloat16Kernel                       # verify + benchmark one baseline
python -m pytest ptx_gym/ptx_gym/kernels/test_triton_kernels.py  # all baselines vs PyTorch
```

## Quick run

Optimize a kernel with the default provider (OpenAI):

```bash
python -m agent MatrixMultiplicationFloat16
```

The agent starts from Triton's own PTX (`triton_generated_ptx/<kernel>.ptx`),
evaluates it, and then applies small PTX patches, evaluating each one. The
best candidate and the full trace land in `output_traces/`.

Switch provider with `--provider`, and model with `--model`:

```bash
python -m agent MatrixMultiplicationFloat16 --provider anthropic
python -m agent MatrixMultiplicationFloat16 --provider openrouter --model qwen/qwen3-coder
```

| Provider | Default model |
| --- | --- |
| `openai` | `gpt-6-astra` |
| `anthropic` | `claude-opus-4-8` |
| `openrouter` | `google/gemini-3.8-flash` |

On OpenRouter, pick a model that supports both `tools` and `response_format`
(see the [model catalog](https://openrouter.ai/models)); `qwen/qwen3-coder`,
`deepseek/deepseek-v3.2`, and `meta-llama/llama-3.3-70b-instruct` work well.

## Agent options

### Starting point

| Flag | Behavior |
| --- | --- |
| *(none)* | Start from Triton's generated PTX. `--start-triton-generated-ptx` is an explicit alias. |
| `--start-json PATH_OR_JSON` | Continue from a saved candidate JSON. Its autotuner metrics are reused, so autotuning is skipped. |
| `--start-ptx PATH` | Edit an existing PTX file with small unified diffs. The result is written to `final_candidate.ptx`. |
| `--initial-prompt-ptx PATH` | Include a reference PTX in the first prompt only; it does not become the working candidate. |

```bash
python -m agent MatrixMultiplicationFloat16 \
  --start-json output_traces/<run>/final_speedup_vs_triton_*.json
```

### Run controls

| Flag | Default | Description |
| --- | --- | --- |
| `--max-tool-rounds` | `3` | Tool-call rounds per model turn. |
| `--max-repair-attempts` | `2` | Repair attempts per failed candidate. |
| `--max-budget` | none | Stop starting new requests once the run's cost reaches this many USD. |
| `--reasoning-effort` | `max` | OpenAI reasoning effort. |
| `--trace-path` | `output_traces` | Where the per-run trace directory is created. |

### Skills and checks

The OpenAI and Anthropic providers upload the bundled PTX and NCU-report
skills before each run; on OpenRouter the model reads the same files through
local tools.

| Flag | Effect |
| --- | --- |
| `--disable-ptx-skill` | Do not provide the PTX ISA reference skill. |
| `--disable-ncu-skill` | Do not provide the Nsight Compute diagnosis skill. |
| `--disable-ncu-report` | Do not profile; plan from source only. |
| `--disable-sanitizer` | Skip `compute-sanitizer` before correctness checks. |

With an NCU report, improvement plans cite only metrics present in the
report. Without one, the planner proposes one to three hypotheses from the
kernel source, PTX, launch configuration, and benchmark results.

## Run outputs

Each run creates a timestamped directory under `--trace-path` containing:

- `triton_generated.ptx`: the PTX Triton compiled at startup.
- Prompts, model responses, and every evaluated candidate as JSON and PTX
  (tool-call candidates under `tool_output/`).
- The final candidate, with its speedup, latency, `tl.constexpr` values, and
  autotuner configuration.
- `prices.log`: API cost, appended as responses arrive, with a total per
  pipeline. Speedup JSONs also record `run_cost_usd_so_far`.

## Utilities

| Command | Purpose |
| --- | --- |
| `python -m extract_ptx` | Dump TTIR, TTGIR, LLVM IR, and PTX for every kernel into `triton_generated_ptx/{ttir,ttgir,llir,ptx}/`. |
| `python -m remeasure_candidate PATH --kernel NAME --output out.json` | Re-run compile, correctness, and timing for a candidate JSON. |
| `python remeasure_baseline.py` | Remeasure archived Triton baselines and write `correction factor.txt` (fresh p50 / archived p50) to each run directory, used by the speedup plot. |

## Validation

```bash
python -m ruff check --exclude ptx_gym .
python -m compileall .
```
