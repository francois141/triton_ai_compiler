<p align="center">
  <img alt="Triton AI Compiler logo" src="logo.png" width="200">
</p>

# Triton AI Compiler

<p align="center">
  <strong>The agent client for TCEnv: it drives an LLM through the loop of
  writing, verifying, and benchmarking PTX for a Triton kernel.</strong>
</p>

<p align="center">
  <img alt="Status: research preview" src="https://img.shields.io/badge/status-research%20preview-orange">
  <img alt="Role: TCEnv client" src="https://img.shields.io/badge/role-TCEnv%20client-blueviolet">
  <img alt="Task: Triton to PTX" src="https://img.shields.io/badge/task-Triton%20%E2%86%92%20PTX-blue">
  <img alt="Hardware: NVIDIA GPU" src="https://img.shields.io/badge/hardware-NVIDIA%20GPU-76B900">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#setup">Setup</a> ·
  <a href="#environment-variables">Environment</a> ·
  <a href="#test-installation">Test installation</a> ·
  <a href="#optimize-a-kernel">Optimize a kernel</a> ·
  <a href="#utilities">Utilities</a> ·
  <a href="#validation">Validation</a> ·
  <a href="#citation">Citation</a>
</p>

## Overview

[TCEnv](ptx_gym/README.md) is the environment: it fixes the compilation
contract for a kernel, evaluates a candidate PTX against it, and reports
correctness, formal verification, and speed. It answers whether a given PTX is
valid and fast, but it does not decide what PTX to try next.

**This repository is the client.** It is an OpenAI-, Anthropic-, and
OpenRouter-compatible tool-calling agent that holds the other half of the loop:
it prompts a model for PTX, submits each candidate to TCEnv, reads back the
verdict and the Nsight Compute report, plans the next edit, and records every
candidate it evaluated.

The two live in separate repositories on purpose. TCEnv is the benchmark and
must stay independent of any particular agent; the client is one agent
implementation among possible others, free to change its prompting, its
provider, and its search strategy without touching the environment it is
measured in. TCEnv is vendored here as the `ptx_gym` submodule, so a
checkout of this repository gives you both halves.

## Setup

### Install

From the shared workspace, initialize the submodules and install every
dependency at once. `uv sync` reads `pyproject.toml`, creates `.venv`, and
installs the OpenAI and Anthropic SDKs, PyTorch, the plotting libraries, and
TCEnv (`ptx_gym`, editable) at the versions pinned in `uv.lock`. The three
exported variables only affect the Triton build: they cap its parallelism and
make `uv` copy instead of hardlink.

```bash
export MAX_JOBS=8
export CMAKE_BUILD_PARALLEL_LEVEL=8
export UV_LINK_MODE=copy
git submodule update --init --recursive
uv sync
```

### Environment variables

One provider key is required, and which one depends on `--provider`. Nothing
else is read from the environment by the agent itself; the remaining variables
are consumed by TCEnv while it evaluates a candidate.

| Variable | Needed for |
| --- | --- |
| `OPENAI_API_KEY` | The default provider (`--provider openai`). |
| `ANTHROPIC_API_KEY` | `--provider anthropic`. |
| `OPENROUTER_API_KEY` | `--provider openrouter`. |
| `OPENAI_BASE_URL` | Optional. Points the OpenAI provider at an OpenAI-compatible endpoint instead of `api.openai.com`. |
| `PTX_MEMORY_SANITIZER` | Path to the `compute-sanitizer` executable. Required unless you pass `--disable-sanitizer`. |
| `NCU_PATH` | Path to the `ncu` executable. Required for profiling; without it, improvement planning falls back to source-only hypotheses. |

A complete setup for the default provider, with both GPU tools present:

```bash
export OPENAI_API_KEY=sk-...
export PTX_MEMORY_SANITIZER=$(which compute-sanitizer)
export NCU_PATH=$(which ncu)
```

Both providers upload the bundled PTX and NCU-report skills before each run.

The OpenRouter harness gives the model on-demand access to the same bundled
PTX and NCU-reference files through local tools. This makes the skills usable
with providers that do not implement proprietary skill-upload APIs.

## Test installation

```bash
.venv/bin/python run_kernel.py SoftmaxFloat16Kernel
python -m pytest ptx_gym/ptx_gym/kernels/test_triton_kernels.py
```

## Optimize a kernel

```bash
python -m agent MatrixMultiplicationFloat16
```

By default, the agent starts from
`triton_generated_ptx/<kernel>.ptx`, evaluates it, and applies localized PTX
patches. It does not generate an initial PTX implementation from scratch.

Use Claude through Anthropic's Messages API by selecting the provider. This
reads `ANTHROPIC_API_KEY`; the OpenAI default reads `OPENAI_API_KEY`.

```bash
python -m agent MatrixMultiplicationFloat16 \
  --provider anthropic
```

Use OpenRouter by setting `OPENROUTER_API_KEY`. Its default model is Gemini
Flash; select any compatible model with `--model`.

```bash
OPENROUTER_API_KEY=... .venv/bin/python -m agent MatrixMultiplicationFloat16 \
  --provider openrouter --model google/gemini-3.8-flash
```

The following OpenRouter model families are useful alternatives when their
selected variant supports both tools and structured JSON output:

```bash
OPENROUTER_API_KEY=... .venv/bin/python -m agent MatrixMultiplicationFloat16 \
  --provider openrouter --model qwen/qwen3-coder
OPENROUTER_API_KEY=... .venv/bin/python -m agent MatrixMultiplicationFloat16 \
  --provider openrouter --model deepseek/deepseek-v3.2
OPENROUTER_API_KEY=... .venv/bin/python -m agent MatrixMultiplicationFloat16 \
  --provider openrouter --model meta-llama/llama-3.3-70b-instruct
```

Check a specific model's `tools` and `response_format` support in the
[OpenRouter model catalog](https://openrouter.ai/models) before running it.

To continue from a candidate JSON file or inline JSON, use `--start-json`:

```bash
python -m agent MatrixMultiplicationFloat16 \
  --start-json output_traces/folder/final_speedup_vs_triton_*.json
```

`--start-triton-generated-ptx` remains available as a compatibility flag; it
has the same behavior as the default:

```bash
python -m agent MatrixMultiplicationFloat16 \
  --start-triton-generated-ptx
```

To optimize an existing PTX file without asking the model to reproduce the
entire file, pass it with `--start-ptx`. The model edits the working PTX using
small unified diffs, each edit is evaluated, and the final source is written to
`final_candidate.ptx` in the run's trace directory.

```bash
python -m agent MatrixMultiplicationFloat16 \
  --start-ptx path/to/candidate.ptx
```

To provide a PTX implementation as reference while still generating a new
initial candidate, use `--initial-prompt-ptx`. Its content is included only in
the first generation prompt; it does not become the working candidate.

```bash
python -m agent MatrixMultiplicationFloat16 \
  --initial-prompt-ptx path/to/reference.ptx
```

Improvement planning uses the current Nsight Compute report when available.
When usable NCU metrics are unavailable, planning falls back to one to three
ideas based on the current kernel source, PTX, launch configuration, and
benchmark results. These ideas label suspected
bottlenecks as hypotheses and require only correctness checks and timing
benchmarks. NCU-backed plans use the bundled NCU report skill and cite only
metrics present in the current report.

Each run creates a timestamped directory under `output_traces/`, containing
prompts, model responses, evaluated candidate JSON and PTX artifacts, and the
final candidate. Speedup JSON artifacts include `run_cost_usd_so_far`, the
cumulative API cost in USD at the time they were written. The final JSON
includes the selected `tl.constexpr` values and autotuner metrics alongside
the final speed and latency. `prices.log` is
appended as API responses arrive and includes a total cost for each completed
agent pipeline. At startup, `triton_generated.ptx` records the PTX compiled by
Triton. Every candidate JSON, including successful PTX tool-call artifacts
under `tool_output/`, embeds the selected autotuning configuration and launch
hyperparameters. Configure the run with
`--provider`, `--model`, `--max-tool-rounds`,
`--max-repair-attempts`, `--reasoning-effort`, and `--trace-path`.
When passed back with `--start-json`, these autotuner metrics are reused and
the kernel skips autotuning.
`--reasoning-effort` applies to OpenAI models; Anthropic requests use the
Messages API's standard tool-use flow.

Use `--disable-ncu-skill`, `--disable-ptx-skill`, `--disable-ncu-report`, or
`--disable-sanitizer` to selectively omit the corresponding uploaded skill or
local validation step. Disabling the NCU report uses the same improvement
planning fallback without profiling.

## Utilities

Extract PTX generated by every Triton kernel:

```bash
python -m extract_ptx
```

The extractor saves TTIR, TTGIR, LLVM IR, and PTX in separate `ttir/`,
`ttgir/`, `llir/`, and `ptx/` directories under `triton_generated_ptx/`.

Re-run compile, correctness, and timing measurement for a candidate JSON:

```bash
python -m remeasure_candidate PATH --kernel KernelClassName \
  --output output.json
```

Remeasure every archived Triton baseline and write the correction factors used
by the speedup plot:

```bash
.venv/bin/python remeasure_baseline.py
```

The command writes `correction factor.txt` to each immediate run directory in
`kernels`. Each factor is the fresh Triton p50 divided by that run's archived
Triton p50, so the plot adjusts its saved speedups to the fresh baseline.

## Validation

```bash
python -m ruff check --exclude ptx_gym .
python -m compileall .
```

## Citation

The environment and this client are both part of the same work:

```bibtex
@inproceedings{aicompiler2027,
  title     = {AI as a Compiler: Compiling Triton Kernels without the Triton Compiler},
  author    = {Anonymous},
  booktitle = {Under review at ICLR},
  year      = {2027}
}
```
