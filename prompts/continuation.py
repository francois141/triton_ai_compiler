from .skills import async_load_store_skill


def build_continuation_prompt(kernel_name: str) -> str:
    """Build the prompt prefix used when continuing from a starting candidate.

    Args:
        kernel_name: Registered Triton PTX kernel class name.

    Returns:
        Compact continuation prompt that avoids asking for an initial candidate.
    """
    return f"""Continue optimizing the verified PTX candidate for {kernel_name}.
Do not restart from the initial kernel prompt or generate a fresh baseline.
Use the current best verified candidate as the source of truth and only propose
targeted changes that preserve the required PTX JSON response schema.

{async_load_store_skill()}""".strip()
