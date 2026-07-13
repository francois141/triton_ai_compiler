from __future__ import annotations

def common_ptxas_issues_skill():
    return """
# Common PTX Issues

- `cvta.to.shared` requires a register operand. Move a shared symbol into a
  `.u64` register before conversion, and use a shared-address form accepted by
  `ptxas` for `cp.async` on the configured target.
- Arithmetic operands must match the instruction type. Convert a `.u32` value
  with `cvt.u64.u32` before using it in `add.u64`.
- Recalculate static shared-memory allocations before emitting PTX. On the
  observed `sm_89` target, keep the total within 49,152 bytes (`0xc000`). Prefer
  smaller tiles or single buffering when necessary.
- PTX entry bodies do not support C-like nested scopes or predicate blocks. Use
  labels, branches, and predicates on individual instructions.
- Control register pressure by reducing per-thread accumulator tiles, reusing
  address temporaries, and avoiding unnecessary `.u64` registers. Inspect
  verbose compiler feedback before tuning further.
- The following syntax is invalid:

@p_warp0 {
    setp.lt.u32 pvalid, rLane, 8;
}

The following syntax is valid

@p_warp0 setp.lt.u32 pvalid, rLane, 8;
""".strip()
