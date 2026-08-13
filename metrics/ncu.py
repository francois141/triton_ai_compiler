import json


def _format_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, ".6g")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _flatten_report(value, path=""):
    if isinstance(value, dict):
        if set(value) == {"value", "unit"}:
            if value["value"] == 0:
                return
            yield path, f"{_format_scalar(value['value'])}{value['unit']}"
            return
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _flatten_report(child, child_path)
        return
    if value != 0:
        yield path, _format_scalar(value)


def _number(report, *path):
    value = report
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _percent(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator * 100


def _first_number(report, paths):
    for path in paths:
        value = _number(report, *path)
        if value is not None:
            return value
    return None


def _metric_number(report, *names):
    metrics = report.get("metrics", {})
    for name in names:
        value = metrics.get(name)
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _derived_ratios(report):
    max_warps = _number(report, "summary", "hardware", "max_warps_per_sm")
    sm_count = _number(report, "summary", "hardware", "sm_count")
    block_dimensions = tuple(
        _number(report, "summary", "kernel", f"block_dim_{axis}") for axis in "xyz"
    )
    grid_dimensions = tuple(
        _number(report, "summary", "kernel", f"grid_dim_{axis}") for axis in "xyz"
    )
    theoretical = _number(report, "summary", "occupancy", "theoretical_pct")
    achieved = _number(report, "summary", "occupancy", "achieved_pct")
    active_warps = _number(report, "summary", "scheduler", "active_warps_per_cycle")
    eligible_warps = _number(report, "summary", "scheduler", "eligible_warps_per_cycle")
    issued_warps = _metric_number(
        report,
        "smsp__issue_active.avg.per_cycle_active",
        "sm__issue_active.avg.per_cycle_active",
    )
    compute_utilization = _first_number(
        report,
        (
            ("summary", "roofline", "compute_peak_pct"),
            ("metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        ),
    )
    dram_utilization = _first_number(
        report,
        (
            ("summary", "roofline", "memory_peak_pct"),
            ("metrics", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
        ),
    )
    l1_utilization = _first_number(
        report,
        (
            ("summary", "memory", "l1_throughput_pct"),
            ("metrics", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
        ),
    )
    l2_utilization = _first_number(
        report,
        (
            ("summary", "memory", "l2_throughput_pct"),
            ("metrics", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        ),
    )
    l1_hit_rate = _first_number(
        report,
        (
            ("summary", "memory", "l1_hit_rate_pct"),
            ("metrics", "l1tex__t_sector_hit_rate.pct"),
        ),
    )
    l2_hit_rate = _first_number(
        report,
        (
            ("summary", "memory", "l2_hit_rate_pct"),
            ("metrics", "lts__t_sector_hit_rate.pct"),
        ),
    )
    dram_read = _first_number(
        report,
        (
            ("summary", "memory", "dram_bytes_read"),
            ("metrics", "dram__bytes_read.sum"),
        ),
    )
    dram_write = _first_number(
        report,
        (
            ("summary", "memory", "dram_bytes_write"),
            ("metrics", "dram__bytes_write.sum"),
        ),
    )
    instructions = _number(report, "summary", "instructions", "executed")
    active_threads = _metric_number(report, "derived__avg_thread_executed_true")
    executed_threads = _metric_number(report, "derived__avg_thread_executed")
    divergent_targets = _metric_number(
        report,
        "smsp__branch_targets_threads_divergent.sum",
        "smsp__sass_branch_targets_threads_divergent.sum",
    )
    branch_instruction_pct = _metric_number(
        report, "derived__smsp__inst_executed_op_branch_pct"
    )
    branch_instructions = _metric_number(report, "smsp__inst_executed_op_branch.sum")
    load_store_instructions = sum(
        _metric_number(report, name) or 0
        for name in (
            "sass__inst_executed_global_loads",
            "sass__inst_executed_global_stores",
            "sass__inst_executed_local_loads",
            "sass__inst_executed_local_stores",
            "sass__inst_executed_shared_loads",
            "sass__inst_executed_shared_stores",
        )
    )
    floating_point_operations = sum(
        _metric_number(report, name) or 0
        for name in (
            "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",
            "derived__sm__sass_thread_inst_executed_op_dfma_pred_on_x2",
            "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2",
            "derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2",
        )
    )
    shared_transactions = sum(
        _number(report, "summary", "memory", name) or 0
        for name in ("shared_load_transactions", "shared_store_transactions")
    )
    shared_bank_conflicts = _number(
        report, "summary", "memory", "shared_bank_conflicts"
    )
    register_count = _number(report, "summary", "kernel", "registers_per_thread")
    max_registers = _metric_number(report, "device__attribute_max_registers_per_thread")
    shared_memory = sum(
        value or 0
        for value in (
            _number(report, "summary", "kernel", "shared_memory_static_bytes"),
            _number(report, "summary", "kernel", "shared_memory_dynamic_bytes"),
        )
    )
    max_shared_memory = _metric_number(
        report,
        "device__attribute_max_shared_memory_per_multiprocessor",
        "device__attribute_max_shared_memory_per_block_optin",
        "device__attribute_max_shared_memory_per_block",
    )
    primary_limit = _number(report, "summary", "occupancy", "limit_warps")

    threads_per_block = None
    if None not in block_dimensions:
        threads_per_block = (
            block_dimensions[0] * block_dimensions[1] * block_dimensions[2]
        )
    grid_blocks = None
    if None not in grid_dimensions:
        grid_blocks = grid_dimensions[0] * grid_dimensions[1] * grid_dimensions[2]

    ratios = {
        "achieved_of_theoretical_occupancy": (_percent(achieved, theoretical), "%"),
        "occupancy_gap": (
            theoretical - achieved
            if theoretical is not None and achieved is not None
            else None,
            "%",
        ),
        "active_warps_of_sm_capacity": (_percent(active_warps, max_warps), "%"),
        "eligible_of_active_warps": (_percent(eligible_warps, active_warps), "%"),
        "eligible_warps_of_sm_capacity": (
            _percent(eligible_warps, max_warps),
            "%",
        ),
        "issued_of_active_warps": (_percent(issued_warps, active_warps), "%"),
        "issued_of_eligible_warps": (_percent(issued_warps, eligible_warps), "%"),
        "launch_blocks_per_sm": (
            grid_blocks / sm_count if grid_blocks is not None and sm_count else None,
            "",
        ),
        "total_launched_warps_per_sm": (
            grid_blocks * threads_per_block / 32 / sm_count
            if grid_blocks is not None and threads_per_block is not None and sm_count
            else None,
            "",
        ),
        "launch_warps_of_sm_capacity": (
            _percent(grid_blocks * threads_per_block / 32, sm_count * max_warps)
            if grid_blocks is not None
            and threads_per_block is not None
            and sm_count
            and max_warps
            else None,
            "%",
        ),
        "launch_waves_at_primary_occupancy_limit": (
            grid_blocks / sm_count / primary_limit
            if grid_blocks is not None and sm_count and primary_limit
            else None,
            "wave",
        ),
        "compute_utilization": (compute_utilization, "%"),
        "dram_utilization": (dram_utilization, "%"),
        "l1_utilization": (l1_utilization, "%"),
        "l2_utilization": (l2_utilization, "%"),
        "compute_to_dram_utilization": (
            compute_utilization / dram_utilization
            if compute_utilization is not None and dram_utilization not in (None, 0)
            else None,
            "x",
        ),
        "l1_hit_rate": (l1_hit_rate, "%"),
        "l2_hit_rate": (l2_hit_rate, "%"),
        "l2_miss_rate": (
            100 - l2_hit_rate if l2_hit_rate is not None else None,
            "%",
        ),
        "dram_read_to_write_bytes": (
            dram_read / dram_write
            if dram_read is not None and dram_write not in (None, 0)
            else None,
            "x",
        ),
        "dram_bytes_per_instruction": (
            (dram_read + dram_write) / instructions
            if dram_read is not None
            and dram_write is not None
            and instructions not in (None, 0)
            else None,
            "byte/inst",
        ),
        "floating_point_operations_per_dram_byte": (
            floating_point_operations / (dram_read + dram_write)
            if floating_point_operations
            and dram_read is not None
            and dram_write is not None
            and dram_read + dram_write
            else None,
            "FLOP/byte",
        ),
        "load_store_instructions_of_total": (
            _percent(load_store_instructions, instructions),
            "%",
        ),
        "active_threads_of_warp": (_percent(active_threads, executed_threads), "%"),
        "divergent_targets_per_branch": (
            divergent_targets / branch_instructions
            if divergent_targets is not None and branch_instructions not in (None, 0)
            else divergent_targets / (instructions * branch_instruction_pct / 100)
            if divergent_targets is not None
            and instructions not in (None, 0)
            and branch_instruction_pct not in (None, 0)
            else None,
            "target/branch",
        ),
        "register_pressure": (_percent(register_count, max_registers), "%"),
        "shared_memory_pressure": (_percent(shared_memory, max_shared_memory), "%"),
        "shared_bank_conflicts_per_transaction": (
            shared_bank_conflicts / shared_transactions
            if shared_bank_conflicts is not None and shared_transactions
            else None,
            "conflict/transaction",
        ),
    }
    for name, limit in (
        ("blocks", _number(report, "summary", "occupancy", "limit_blocks")),
        ("registers", _number(report, "summary", "occupancy", "limit_registers")),
        (
            "shared_memory",
            _number(report, "summary", "occupancy", "limit_shared_memory"),
        ),
        ("warps", _number(report, "summary", "occupancy", "limit_warps")),
    ):
        ratios[f"occupancy_limit_{name}_of_primary"] = (
            _percent(primary_limit, limit),
            "%",
        )
    return {
        name: f"{_format_scalar(value)}{unit}"
        for name, (value, unit) in ratios.items()
        if value not in (None, 0)
    }


def compact_ncu_report(ncu_report):
    report_lines = [
        f"derived.{name}: {value}"
        for name, value in _derived_ratios(ncu_report).items()
    ]
    report_lines.extend(
        f"{path}: {value}"
        for path, value in _flatten_report(ncu_report)
        if path not in {"available", "error", "return_code", "source_report"}
    )
    return "\n".join(report_lines)
