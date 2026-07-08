# Triton PTX Client

Client commands for prompt generation, PTX extraction, evaluation, verification, and test-time scaling.

This folder should live next to the `triton_ptx` repository in a shared
workspace directory:

```text
<workspace>/
  client/
  triton_ptx/
```

## Setup

From the shared workspace directory, create a virtual environment, install the
`triton_ptx` codebase in editable mode, and make the top-level `client` package
importable:

```bash
export WORKSPACE_ROOT=/path/to/workspace
cd "$WORKSPACE_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e triton_ptx
export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
```

The editable install reads `triton_ptx/requirements.txt`, which is also wired
through `triton_ptx/pyproject.toml` and `setup.py`.

Other codebases can use the library API after installation:

```python
from triton_ptx import Payload, TritonPTXCandidateEvaluator, resolve_kernel

kernel_cls = resolve_kernel("AddKernel")
evaluator = TritonPTXCandidateEvaluator(kernel_cls)
result = evaluator.evaluate(Payload(ptx="...", threads_x=256))
```

## Common Commands

Run all tests and verification checks:

```bash
cd "$WORKSPACE_ROOT"
python3 -m pytest triton_ptx/triton_ptx client
```

Run test-time scaling for one kernel:

```bash
python3 -m test_time_scaling_loop AddKernel --config configs/test_time_scaling_openai.yaml
```

Run the deterministic fake AddKernel test-time-scaling loop:

```bash
python3 -m test_time_scaling_loop AddKernel --config configs/test_time_scaling_fake_add.yaml
```

Run an OpenAI tool-calling agent loop where the model can directly call local
compile, correctness, and benchmark tools:

```bash
python3 -m openai_agent_tools MatrixMultiplicationKernel
```

The complete response trace is written to `trace.json`, while the final
candidate and its measured speedup are written beside it as `trace_final.json`.
To continue
optimizing an existing candidate, pass either its file or inline JSON:

```bash
python3 -m openai_agent_tools MatrixMultiplicationKernel \
  --start-json trace_final.json
```

Run test-time scaling against Anthropic Claude Opus 4.8:

```bash
python3 -m test_time_scaling_loop AddKernel --config configs/test_time_scaling_anthropic.yaml
```

Run test-time scaling against Gemini 2.5 Pro:

```bash
python3 -m test_time_scaling_loop AddKernel --config configs/test_time_scaling_gemini.yaml
```


Run test-time scaling against Openrouter: 

```bash
python3 -m test_time_scaling_loop AddKernel --config configs/test_time_scaling_openrouter.yaml
```

Write test-time scaling artifacts to a custom database directory:

```bash
python3 -m test_time_scaling_loop AddKernel --config path/to/config.yaml
```

The test-time scaling command expects the kernel class name first and an
optional OmegaConf YAML path via `--config`. If `--config` is omitted, the
in-file default config in `test_time_scaling_loop.py` is used.

The config controls loop parameters, generator selection, model settings, and
the archive directory:

```yaml
loop:
  rounds: 10
  k: 2
  max_retries: 3

generator:
  provider: openai
  model: gpt-5
  options:
    reasoning_effort: medium

storage:
  database_dir: database
```

Ready-to-edit presets live in `configs/`:

- `test_time_scaling_openai.yaml`
- `test_time_scaling_anthropic.yaml`
- `test_time_scaling_gemini.yaml`
- `test_time_scaling_openrouter.yaml`

The OpenRouter preset expects `OPENROUTER_API_KEY` in the environment. The
`openrouter` provider includes a small price table for these model IDs:
`qwen/qwen3-coder` and `deepseek/deepseek-v4-pro`.

Extract embedded PTX from Triton kernels:

```bash
python3 -m extract_ptx
```

Summarize archived winners from `database/`:

```bash
python3 -m measure_ptx --database-dir /path/to/database
```

Re-run compile, correctness, and benchmark measurement for one archived result:

```bash
python3 -m remeasure_candidate PATH --output output.json
```

Lint the package:

```bash
cd "$WORKSPACE_ROOT"
python3 -m ruff check triton_ptx/triton_ptx
python3 -m ruff check --fix triton_ptx/triton_ptx
```

## Notes

- Test-time scaling archives each run under `database/<timestamp>_<kernel>/`.
- `measure_ptx` reads archived `output_winner_*.json` files and reports the best valid `speedup_vs_triton` found for each kernel.
