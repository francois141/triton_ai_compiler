#!/usr/bin/env bash
#
# Re-runs `volta analyze` for every astra-corpus kernel that has PTX to
# analyze, using the exact ground-truth launch configs documented in
# report.md's per-kernel sections. Companion to report.md.
#
# `.spec` files now exist (in `specs/`, same naming convention as
# `paper_results_2/specs/`) for SiLU, SwiGLU, and ReLU — their candidates
# also get a `volta verify` pass against the exact mathematical ground
# truth, not just `analyze`'s race/deadlock check. GELU has no spec (the
# spec DSL has no `erf` primitive, and its ground truth is exact erf-based
# GELU) and MatrixMultiplicationFloat16's reference side has an unresolved
# prologue race (see `triton_prologue_race_investigation.md`) blocking any
# `verify` there, so neither gets a verify step here even though
# `specs/` does hold a matmul spec (used manually against the candidate
# only — see that investigation doc and the session transcript).
#
# NOTE: report.md predates this session's `tensormap.*`/`cp.async.bulk.
# tensor`/`tcgen05.mma` (multi-operand-shape) and `copysign` lowering work
# landing — its GELU and MatrixMultiplicationFloat16 sections describing
# both as blocked/failing-to-parse are now stale (GELU's candidate clears
# `analyze`; MatrixMultiplicationFloat16's candidate clears both `analyze`
# and `verify` against `specs/`, and its reference side reaches the race
# above instead of failing to parse). Not corrected here or in report.md -
# flag to whoever asked for this script to run if it matters for your use.
#
# Each kernel here has no `final_candidate.ptx`; this script analyzes the
# highest-numbered `tool_output/*_launch_verifier*.ptx` file in each run
# directory (see report.md's "Naming convention" section for why).
#
# Usage: ./rerun_analysis.sh [--no-build]
#   --no-build   skip `cargo build --release` and use whatever
#                target/release/volta already exists (errors if missing).
#
# Env overrides:
#   VOLTA_REPO   path to the volta repo checkout
#                (default: ../../volta relative to this script)
#   Z3_LIB_DIR   directory containing libz3
#                (default: /opt/homebrew/opt/z3/lib)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOLTA_REPO=$SCRIPT_DIR/volta

echo $VOLTA_REPO
Z3_LIB_DIR="${Z3_LIB_DIR:-/opt/homebrew/opt/z3/lib}"

if [[ -z "$VOLTA_REPO" || ! -d "$VOLTA_REPO" ]]; then
  echo "error: could not find the volta repo checkout — set VOLTA_REPO explicitly" >&2
  exit 1
fi

VOLTA_BIN="$VOLTA_REPO/target/release/volta"
export DYLD_LIBRARY_PATH="$Z3_LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

if [[ "${1:-}" != "--no-build" ]]; then
  echo "Building volta_cli (release)..."
  ( cd "$VOLTA_REPO" && RUSTFLAGS="-L $Z3_LIB_DIR" cargo build -p volta_cli --release ) || {
    echo "error: build failed" >&2
    exit 1
  }
  echo
fi

if [[ ! -x "$VOLTA_BIN" ]]; then
  echo "error: $VOLTA_BIN not found (build it first, or drop --no-build)" >&2
  exit 1
fi

echo "volta binary: $VOLTA_BIN"
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
echo

RESULTS=()

# do_analyze <dir> <label> <ptx-file> <analyze-args...>
# Returns: 0 clean pass, 1 failed/blocked (lowering/parse error, NotConcrete,
# race, deadlock, etc.), 2 file absent.
do_analyze() {
  local dir="$1" label="$2" ptx_file="$3"
  shift 3
  local path="$SCRIPT_DIR/$dir/$ptx_file"
  echo "----- $label — analyze $(basename "$ptx_file") -----"
  if [[ ! -f "$path" ]]; then
    echo "  (missing: $ptx_file)"
    return 2
  fi
  local out
  out=$("$VOLTA_BIN" analyze "$path" --no-log-file --no-profile --print-outputs 0 "$@" 2>&1)
  printf '%s\n' "$out" | sed 's/^/  /'
  grep -q '^Analysis complete' <<<"$out" && return 0
  return 1
}

# run_kernel <dir> <candidate-file> <triton-file-or-empty> <label> <args...>
run_kernel() {
  local dir="$1" candidate="$2" triton="$3" label="$4"
  shift 4

  do_analyze "$dir" "$label (candidate)" "$candidate" "$@"
  local c_rc=$? c_status
  case $c_rc in
    0) c_status="CLEAN PASS" ;;
    2) c_status="ABSENT" ;;
    *) c_status="FAILED/BLOCKED" ;;
  esac

  local t_status="n/a"
  if [[ -n "$triton" ]]; then
    do_analyze "$dir" "$label (triton)" "$triton" "$@"
    local t_rc=$?
    case $t_rc in
      0) t_status="CLEAN PASS" ;;
      2) t_status="ABSENT" ;;
      *) t_status="FAILED/BLOCKED" ;;
    esac
  fi

  RESULTS+=("${label}|${c_status}|${t_status}")
  echo
}

VERIFY_RESULTS=()

# do_verify <dir> <label> <ptx-file> <spec-file> <verify-args...>
# Returns: 0 EQUIVALENT, 1 failed/blocked/counterexample, 2 file absent.
do_verify() {
  local dir="$1" label="$2" ptx_file="$3" spec_file="$4"
  shift 4
  local path="$SCRIPT_DIR/$dir/$ptx_file"
  local spec_path="$SCRIPT_DIR/specs/$spec_file"
  echo "----- $label — verify $(basename "$ptx_file") against specs/$spec_file -----"
  if [[ ! -f "$path" ]]; then
    echo "  (missing: $ptx_file)"
    return 2
  fi
  if [[ ! -f "$spec_path" ]]; then
    echo "  (missing spec: $spec_file)"
    return 2
  fi
  local out
  out=$("$VOLTA_BIN" verify "$path" "$spec_path" --no-log-file --no-profile "$@" 2>&1)
  printf '%s\n' "$out" | sed 's/^/  /'
  grep -q '^EQUIVALENT' <<<"$out" && return 0
  return 1
}

# run_verify <dir> <candidate-file> <spec-file> <label> <verify-args...>
# Candidate only — no `.spec` currently applies to a triton reference side
# in this corpus (see the header comment for why).
run_verify() {
  local dir="$1" candidate="$2" spec_file="$3" label="$4"
  shift 4

  do_verify "$dir" "$label (candidate)" "$candidate" "$spec_file" "$@"
  local rc=$? status
  case $rc in
    0) status="EQUIVALENT" ;;
    2) status="ABSENT" ;;
    *) status="FAILED/BLOCKED" ;;
  esac

  VERIFY_RESULTS+=("${label}|${status}")
  echo
}

# --- kernels -------------------------------------------------------------
# Flags below are copied verbatim from report.md's per-kernel sections — do
# not "fix" a value here without also correcting report.md.


run_verify \
  "astra/260918022932_SiLUFloat8Kernel_gpt-6-astra_medium" \
  "iteration_000_candidate_00_try_00_initial_candidate_speedup_vs_triton_0.9931x.ptx" \
  "silu.spec" \
  "SiLU" \
  -k kernel -b 128 -g 1 --dyn-shared 4 \
  --array x:0x100000000:1:134217728:in --array output:0x200000000:2:134217728:out \
  --param ptr:x --param ptr:output --param int:0 --param int:0 \
  --dim N=134217728 --sample 256 --verify-numeric



# --- summary -------------------------------------------------------------

echo "===================== SUMMARY ====================="
printf '%-32s %-16s %-16s\n' "Kernel" "candidate" "triton"
printf '%-32s %-16s %-16s\n' "------" "---------" "------"
for r in "${RESULTS[@]}"; do
  IFS='|' read -r label c t <<<"$r"
  printf '%-32s %-16s %-16s\n' "$label" "$c" "$t"
done
echo
echo "6. FusedGEMMAddGELUFloat16Kernel is not listed above (no PTX yet)."
echo
echo "Compare this table against report.md's results table — flag anything"
echo "that disagrees (note: report.md is stale for GELU and"
echo "MatrixMultiplicationFloat16 — see the header comment above)."
echo

echo "================= VERIFY SUMMARY (candidate vs. specs/) ================="
printf '%-32s %-16s\n' "Kernel" "verify"
printf '%-32s %-16s\n' "------" "------"
for r in "${VERIFY_RESULTS[@]}"; do
  IFS='|' read -r label v <<<"$r"
  printf '%-32s %-16s\n' "$label" "$v"
done
echo
echo "GELU and MatrixMultiplicationFloat16 have no verify row: GELU's exact"
echo "ground truth needs erf, unavailable in the spec DSL; MatrixMultiplication's"
echo "reference side has an unresolved race (see"
echo "triton_prologue_race_investigation.md) and was never wired into this"
echo "script's verify pass even though specs/ holds a matmul spec."







