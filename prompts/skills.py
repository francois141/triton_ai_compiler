from __future__ import annotations


def async_load_store_skill() -> str:
    """Return guidance for valid asynchronous PTX loads and global stores.

    Returns:
        Markdown prompt text describing async load and store requirements.
    """
    return """"""
    return """
# Async Load and Store Skills

Use `cp.async` only for global-to-shared loads on `sm_89`. Use normal
`st.global` instructions for global stores; PTX does not provide a matching
asynchronous global-store operation on this target.

When using:

```ptx
cp.async.ca.shared.global [dst], [src], N;
```

follow these requirements:

- The global source address must be valid for the entire transfer. For a
  16-byte copy, bytes `[src, src + 15]` must all be accessible; checking only
  the first element is insufficient.
- Handle boundary tiles with scalar predicated loads, 4-byte async copies, or
  the zero-fill (`ignore-src` or `src-size`) variants. Never issue an async copy
  from an invalid global address.
- Keep the complete destination range inside the allocated shared-memory
  region.
- A 16-byte copy requires 16-byte alignment. For FP32 matrices, the base
  pointer, row stride, and vectorized column offset must preserve that
  alignment. Otherwise, use scalar loads or 4-byte copies.
- Every thread that issues an async copy must execute the matching group
  operations before any thread consumes the shared data:

```ptx
cp.async.commit_group;
cp.async.wait_group 0;
```

Use the 16-byte path only when alignment and full-vector bounds are guaranteed.

###`cvta.to` (Convert Address)

The `cvta.to` instruction converts a **generic address** into an address belonging to a specific PTX memory space (`.global`, `.shared`, `.local`, or `.const`).

```ptx
cvta.to.global.u64  dst, src;
cvta.to.shared.u64  dst, src;
cvta.to.local.u64   dst, src;
cvta.to.const.u64   dst, src;
```

- `dst`: destination register (typically `.u64`)
- `src`: source generic address
- `.global`, `.shared`, `.local`, `.const`: target memory space

## Example

```ptx
.reg .u64 %rd1, %rd2;

cvta.to.global.u64 %rd2, %rd1;
ld.global.u32 %r1, [%rd2];
```

In particular, this is wrong

.shared .align 16 .b8 smemA[16384];
cvta.to.shared.u64 sA_base, smemA; // Not a generic pointer here
""".strip()


def common_ptxas_issues_skill() -> str:
    """Return guidance for preventing common PTXAS failures.

    Returns:
        Markdown prompt text listing common PTXAS issues and mitigations.
    """
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
