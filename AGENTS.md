# Agent Guidelines for Python Code Quality

This document provides guidelines for maintaining high-quality Python code. These rules MUST be followed by all AI coding agents and contributors.

## Your Core Principles

All code you write MUST be fully optimized.

"Fully optimized" includes:

* maximizing algorithmic big-O efficiency for memory and runtime
* using parallelization and vectorization where appropriate
* following proper style conventions for the code language, including maximizing code reuse and following DRY principles
* adding no extra code beyond what is absolutely necessary to solve the user's problem
* using a small, low-overhead Python library when it significantly reduces the amount of new code needed while maintaining optimal performance

If the code is not fully optimized before handing it to the user, perform another optimization and quality pass.

## Kernel Files

Each kernel file under `ptx_gym/ptx_gym/kernels/` is a self-contained benchmark
case. This rule overrides the DRY and code-reuse principles above.

* Never compress, deduplicate, or refactor code within a kernel file or across kernel
  files, even when several kernels look alike.
* Never move kernel code or `@triton.jit` helpers into a shared module. The prompt
  builder sends the kernel source to the model verbatim through `inspect.getsource`,
  and it only includes helpers that are defined in the kernel's own module.
* Never run `ruff format` or other rewriting tools on kernel files. Any change to the
  kernel source changes the prompt and invalidates comparisons with earlier runs.
* Fix only real defects in kernel files, such as a wrong reference implementation,
  input generation, launch configuration, or tolerance, and keep each fix minimal.

## Preferred Tools

* Use `uv` for Python package management and to create a `.venv` if it is not present.
* Ensure `ipykernel` and `ipywidgets` are installed in `.venv` for Jupyter Notebook compatibility. Do not include them in project package requirements unless the application directly depends on them.
* Use `tqdm` to track long-running loops within Jupyter Notebooks. The progress-bar description must be contextually relevant.
* Use `orjson` for JSON loading and dumping when its performance or features are beneficial.
* When reporting errors to the console, use `logger.error` instead of `print`.
* If the project creates images such as PNG or WebP files, verify that the rendered images meet the user's and application's requirements.
* For data science:

  * Always use `polars` instead of `pandas` for DataFrame manipulation.
  * When printing a Polars DataFrame, do not simultaneously print its row count or schema unless that information is independently necessary.
  * Inspect data in subsets of no more than 10 rows at a time.
* For database creation:

  * Do not denormalize unless explicitly requested.
  * Use the most appropriate data type, such as `DATETIME` or `TIMESTAMP` for datetime-related fields.
  * Use native array or nested data types for nested fields when the selected database supports them. Do not serialize structured values into text unless required for compatibility.
* In Jupyter Notebooks, explicitly call `print()` for objects inside conditional blocks when notebook display behavior would otherwise suppress the output.

## Code Style and Formatting

* Use meaningful, descriptive variable and function names.
* Follow PEP 8 style guidelines.
* Use four spaces for indentation. Never use tabs.
* Do not use emoji or Unicode characters that emulate emoji. Multibyte-character test data is an exception when it is relevant to the task.
* Use `snake_case` for functions and variables.
* Use `PascalCase` for classes.
* Use `UPPER_CASE` for constants.
* Limit line length to 88 characters, following Ruff's default formatter convention.
* Avoid redundant comments that merely repeat what the code already expresses.
* Do not include comments that expose the original prompt or unrelated implementation instructions.
* Keep comments accurate when the code changes.

## Error Handling

* Never silently swallow exceptions.
* Never use bare `except:` clauses.
* Catch specific exception types rather than broad exception types.
* Log failures when logging is appropriate.
* Use context managers for resources that require cleanup.
* Provide meaningful error messages with enough context to diagnose the failure.
* Avoid exposing secrets, tokens, credentials, or sensitive personal information in error messages.

## Function Design

* Keep functions focused on a single responsibility.
* Never use mutable objects such as lists or dictionaries as default argument values.
* Limit functions to five parameters or fewer when practical.
* Return early when doing so reduces unnecessary nesting.
* Avoid unnecessary helper functions that do not improve reuse, clarity, or separation of responsibilities.
* Prefer iterators and generators when they reduce memory usage without harming readability.

## Class Design

* Keep classes focused on a single responsibility.
* Keep `__init__` simple and avoid complex initialization logic.
* Use dataclasses for simple data containers when they reduce boilerplate.
* Prefer composition over inheritance.
* Do not add methods that are not necessary.
* Use `@property` for computed attributes when attribute-style access is appropriate.
* Avoid creating a class when a small set of focused functions is sufficient.

## Verification

* Do not create automated tests unless the user explicitly requests them.
* Validate the implementation using the smallest appropriate set of checks.
* Do not modify or manipulate benchmarks or validation criteria to produce misleading results.
* Save any explicitly requested test code in separate files before running it.
* Never delete files created as part of an explicitly requested testing process.
* Ensure directories used for generated test outputs are included in `.gitignore`.
* Mock external dependencies such as APIs, databases, and file systems when tests are explicitly requested.
* Do not leave commented-out checks, temporary debugging code, or breakpoints in the final implementation.

## Imports and Dependencies

* Avoid wildcard imports such as `from module import *`.
* Document runtime dependencies in `pyproject.toml`.
* Use `uv` for package management and dependency resolution.
* Organize imports in this order:

  1. standard-library imports
  2. third-party imports
  3. local imports
* Use Ruff to sort and format imports.
* Avoid adding dependencies when the standard library provides an equally clear and efficient solution.
* Remove unused dependencies.

## Python Best Practices

* Never use mutable default arguments.
* Use context managers for files, connections, locks, and other managed resources.
* Use `is` and `is not` for comparisons with `None`.
* Use direct boolean checks where appropriate instead of comparing values with `True` or `False`.
* Use f-strings for string interpolation.
* Use list comprehensions and generator expressions when they improve clarity.
* Use `enumerate()` instead of manually maintaining counter variables.
* Use built-in functions and standard-library tools when they provide clear, optimized implementations.
* Avoid unnecessary intermediate collections.
* Avoid premature parallelization when task size or overhead makes sequential execution faster.
* Ensure multiprocessing and threading are appropriate for the workload and execution environment before using them.

## Benchmarking and Optimization

* Never run benchmarks in parallel because competing workloads can invalidate the results.
* Never manipulate benchmarks to satisfy performance requirements artificially.
* Ensure comparisons between libraries or implementations use equivalent workloads, inputs, configurations, and output validation.
* Keep benchmark cases independent.
* Disable caching when it would make supposedly independent benchmark cases dependent.
* Measure before introducing complex optimizations.
* Consider runtime, memory usage, startup overhead, maintainability, and expected input size.
* Do not sacrifice correctness or clarity for negligible performance gains.

## Security

* Never store secrets, API keys, or passwords in source code.
* Store sensitive local configuration in environment variables or an appropriate secret-management system.
* Declare `.env` in `.gitignore`.
* Never print or log URLs containing API keys or other credentials.
* Never log passwords, tokens, session identifiers, private keys, or sensitive personal information.
* Validate untrusted input at system boundaries.
* Use parameterized queries for database operations.
* Avoid executing dynamically constructed code or shell commands.
* When shell execution is necessary, pass arguments as a sequence instead of constructing a shell string.
* Apply the principle of least privilege to files, credentials, services, and database accounts.

## Version Control

* Write clear, descriptive commit messages.
* Never commit commented-out code.
* Never commit debug print statements or breakpoints.
* Never commit credentials or sensitive data.
* Keep commits focused on one logical change when possible.
* Do not include generated files unless they are required project artifacts.

## Tools

* Use Ruff for code formatting, linting, and import sorting.
* Use `uv` for package management.
* Use the project's existing validation tools when available.
* Do not introduce mypy, pytest, or another validation dependency unless the user explicitly requests it or the existing project already requires it.

## Python Environment

* Always use `.venv/bin/python` for Python commands.
* Do not use `python` or the system `python3`.
* Use `.venv/bin/python -m compileall` for syntax validation.
* Run package tools through `.venv/bin/python -m <tool>` when the tool supports module execution.
* Use `uv` to create or update the virtual environment.
* Reuse an existing valid `.venv` rather than recreating it unnecessarily.

## Before Committing

* [ ] The implementation satisfies the requested behavior.
* [ ] Relevant existing project checks pass.
* [ ] Syntax validation passes.
* [ ] Ruff formatting and linting pass.
* [ ] No unnecessary code or dependencies were added.
* [ ] No commented-out code, debug statements, or breakpoints remain.
* [ ] No credentials or sensitive information are hardcoded.
* [ ] Resource handling and error paths have been reviewed.
* [ ] Performance is appropriate for the expected workload.

---

**Remember:** Prioritize correctness, clarity, maintainability, and measured performance over cleverness. Do not generate tests, type annotations, or docstrings unless the user explicitly requests them.
