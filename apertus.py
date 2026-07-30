import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as functional
import triton
import triton.language as tl
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_ID = "swiss-ai/Apertus-v1.5-8B"
MODEL_CACHE_DIR = "/matx/u/franc141/huggingface-cache"
MODEL_WEIGHTS_DIR = Path(
    "/matx/u/franc141/huggingface-cache/models--swiss-ai--Apertus-v1.5-8B/"
    "snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6"
)
VOCAB_SIZE = 266_752
OUTPUT_VOCAB_SIZE = 131_072
HIDDEN_SIZE = 4_096
INTERMEDIATE_SIZE = 21_504
NUM_LAYERS = 32
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
RMS_NORM_EPS = 1e-5
ROPE_THETA = 4_000_000
ROPE_FACTOR = 32.0
ROPE_LOW_FREQ_FACTOR = 1.0
ROPE_HIGH_FREQ_FACTOR = 4.0
ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS = 8_192
EOS_TOKEN_IDS = (2, 68, 72)
RMS_NORM_BLOCK_SIZE = 128
XIELU_BLOCK_SIZE = 256
ROPE_BLOCK_SIZE = 128
ATTENTION_BLOCK_M = 32
ATTENTION_BLOCK_N = 64


@triton.jit
def _linear_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_features: tl.constexpr,
    input_row_stride: tl.constexpr,
    weight_output_stride: tl.constexpr,
    weight_input_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program_m = tl.program_id(0)
    program_n = tl.program_id(1)
    row_offsets = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_offsets = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    input_offsets = tl.arange(0, BLOCK_K)
    input_ptrs = (
        input_ptr
        + row_offsets[:, None] * input_row_stride
        + input_offsets[None, :]
    )
    weight_ptrs = (
        weight_ptr
        + output_offsets[None, :] * weight_output_stride
        + input_offsets[:, None] * weight_input_stride
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, input_features // BLOCK_K):
        inputs = tl.load(input_ptrs)
        weights = tl.load(weight_ptrs)
        accumulator = tl.dot(inputs, weights, acc=accumulator)
        input_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K * weight_input_stride

    output_ptrs = (
        output_ptr
        + row_offsets[:, None] * output_row_stride
        + output_offsets[None, :]
    )
    tl.store(output_ptrs, accumulator.to(output_ptr.dtype.element_ty))


def triton_linear(hidden_states, weight):
    input_features = hidden_states.shape[-1]
    output_features, weight_input_features = weight.shape
    if input_features != weight_input_features:
        raise ValueError(
            f"Input features ({input_features}) do not match weight features "
            f"({weight_input_features})."
        )
    assert input_features % 32 == 0, "Input features must be divisible by 32."
    assert output_features % 128 == 0, "Output features must be divisible by 128."

    if (
        not hidden_states.is_cuda
        or not weight.is_cuda
        or hidden_states.dtype not in (torch.float16, torch.bfloat16)
        or torch.is_grad_enabled()
    ):
        return functional.linear(hidden_states, weight)

    flattened_input = hidden_states.reshape(-1, input_features).contiguous()
    num_rows = flattened_input.shape[0]
    output = torch.empty(
        (num_rows, output_features),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    grid = (
        num_rows,
        triton.cdiv(output_features, 128),
    )
    _linear_kernel[grid](
        flattened_input,
        weight,
        output,
        input_features,
        flattened_input.stride(0),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        BLOCK_M=1,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=4,
        num_stages=4,
    )
    return output.reshape(*hidden_states.shape[:-1], output_features)


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_offset = row * hidden_size
    squared_sum = 0.0
    for block_offset in range(0, hidden_size, BLOCK_SIZE):
        values = tl.load(input_ptr + row_offset + block_offset + offsets).to(
            tl.float32
        )
        squared_sum += tl.sum(values * values, axis=0)

    variance = squared_sum / hidden_size
    inverse_rms = tl.rsqrt(variance + eps)
    for block_offset in range(0, hidden_size, BLOCK_SIZE):
        values = tl.load(input_ptr + row_offset + block_offset + offsets).to(
            tl.float32
        )
        normalized = (values * inverse_rms).to(input_ptr.dtype.element_ty)
        weights = tl.load(weight_ptr + block_offset + offsets)
        tl.store(
            output_ptr + row_offset + block_offset + offsets,
            normalized * weights,
        )


def triton_rms_norm(hidden_states, weight, eps):
    hidden_size = hidden_states.shape[-1]
    assert hidden_states.is_cuda, "RMSNorm requires CUDA input."
    assert weight.is_cuda, "RMSNorm requires CUDA weights."
    assert hidden_states.device == weight.device, (
        "RMSNorm input and weights must be on the same device."
    )
    assert weight.is_contiguous(), "RMSNorm weights must be contiguous."
    assert weight.shape == (hidden_size,), (
        f"RMSNorm weights must have shape ({hidden_size},)."
    )
    assert weight.dtype == hidden_states.dtype, (
        "RMSNorm weights must match the input dtype."
    )
    assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
        "RMSNorm input must use float16 or bfloat16."
    )
    assert not torch.is_grad_enabled(), "RMSNorm does not support autograd."

    hidden_states = hidden_states.contiguous()
    flattened_input = hidden_states.reshape(-1, hidden_size)
    assert hidden_size % RMS_NORM_BLOCK_SIZE == 0, (
        "RMSNorm hidden size must be divisible by "
        f"{RMS_NORM_BLOCK_SIZE}; received {hidden_size}."
    )
    output = torch.empty_like(flattened_input)
    _rms_norm_kernel[(flattened_input.shape[0],)](
        flattened_input,
        weight,
        output,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=RMS_NORM_BLOCK_SIZE,
        num_warps=4,
    )
    return output.reshape_as(hidden_states)


class Apertus1p5TextRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=RMS_NORM_EPS, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        self.eps = eps

    def forward(self, hidden_states):
        return triton_rms_norm(hidden_states, self.weight, self.eps)


@triton.jit
def _xielu_kernel(
    input_ptr,
    alpha_p_ptr,
    alpha_n_ptr,
    beta_ptr,
    eps_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(axis=0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets).to(tl.float32)
    alpha_p = tl.load(alpha_p_ptr).to(tl.float32)
    alpha_n = tl.load(alpha_n_ptr).to(tl.float32)
    beta = tl.load(beta_ptr).to(tl.float32)
    eps = tl.load(eps_ptr).to(tl.float32)
    positive_alpha = tl.log(1.0 + tl.exp(alpha_p))
    negative_alpha = beta + tl.log(1.0 + tl.exp(alpha_n))
    positive = positive_alpha * values * values + beta * values
    negative = (tl.exp(tl.minimum(values, eps)) - 1.0 - values) * negative_alpha
    negative += beta * values
    tl.store(output_ptr + offsets, tl.where(values > 0.0, positive, negative))


def triton_xielu(hidden_states, alpha_p, alpha_n, beta, eps):
    assert hidden_states.is_cuda, "XIELU requires CUDA input."
    assert hidden_states.is_contiguous(), "XIELU input must be contiguous."
    assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
        "XIELU input must use float16 or bfloat16."
    )
    assert alpha_p.device == hidden_states.device, (
        "alpha_p must be on the input device."
    )
    assert alpha_n.device == hidden_states.device, (
        "alpha_n must be on the input device."
    )
    assert beta.device == hidden_states.device, "beta must be on the input device."
    assert eps.device == hidden_states.device, "eps must be on the input device."
    assert alpha_p.dtype == hidden_states.dtype, "alpha_p must match the input dtype."
    assert alpha_n.dtype == hidden_states.dtype, "alpha_n must match the input dtype."
    assert beta.dtype == hidden_states.dtype, "beta must match the input dtype."
    assert eps.dtype == hidden_states.dtype, "eps must match the input dtype."
    assert alpha_p.numel() == 1, "alpha_p must be a scalar tensor."
    assert alpha_n.numel() == 1, "alpha_n must be a scalar tensor."
    assert beta.numel() == 1, "beta must be a scalar tensor."
    assert eps.numel() == 1, "eps must be a scalar tensor."
    assert hidden_states.numel() > 0, "XIELU input must not be empty."
    assert not torch.is_grad_enabled(), "XIELU does not support autograd."

    flattened_input = hidden_states.reshape(-1)
    assert flattened_input.numel() % XIELU_BLOCK_SIZE == 0, (
        "XIELU input size must be divisible by "
        f"{XIELU_BLOCK_SIZE}; received {flattened_input.numel()} elements."
    )
    output = torch.empty_like(flattened_input)
    _xielu_kernel[(flattened_input.numel() // XIELU_BLOCK_SIZE,)](
        flattened_input,
        alpha_p,
        alpha_n,
        beta,
        eps,
        output,
        BLOCK_SIZE=XIELU_BLOCK_SIZE,
        num_warps=4,
    )
    return output.reshape_as(hidden_states)


class XIELUActivation(torch.nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.alpha_p = torch.nn.Parameter(
            torch.log(torch.expm1(torch.tensor(0.8, device=device, dtype=dtype)))
            .unsqueeze(0)
        )
        self.alpha_n = torch.nn.Parameter(
            torch.log(torch.expm1(torch.tensor(0.3, device=device, dtype=dtype)))
            .unsqueeze(0)
        )
        self.register_buffer("beta", torch.tensor(0.5, device=device, dtype=dtype))
        self.register_buffer("eps", torch.tensor(-1e-6, device=device, dtype=dtype))

    def forward(self, hidden_states):
        return triton_xielu(
            hidden_states,
            self.alpha_p,
            self.alpha_n,
            self.beta,
            self.eps,
        )

@triton.jit
def _rope_kernel(
    input_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    sequence_length: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    feature_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    is_first_half = feature_offsets < half_head_dim
    paired_offsets = tl.where(
        is_first_half,
        feature_offsets + half_head_dim,
        feature_offsets - half_head_dim,
    )
    input_offsets = row * head_dim + feature_offsets
    paired_values = tl.load(input_ptr + row * head_dim + paired_offsets)
    values = tl.load(input_ptr + input_offsets)
    rotated_values = tl.where(is_first_half, -paired_values, paired_values)
    position = row % sequence_length
    rope_offsets = position * head_dim + feature_offsets
    cos_values = tl.load(cos_ptr + rope_offsets)
    sin_values = tl.load(sin_ptr + rope_offsets)
    output = values * cos_values + rotated_values * sin_values
    tl.store(output_ptr + input_offsets, output)


def apply_rotary_embedding(hidden_states, cos, sin):
    batch_size, num_heads, sequence_length, head_dim = hidden_states.shape
    assert hidden_states.is_cuda, "RoPE requires CUDA input."
    assert hidden_states.is_contiguous(), "RoPE input must be contiguous."
    assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
        "RoPE input must use float16 or bfloat16."
    )
    assert cos.is_cuda and sin.is_cuda, "RoPE frequencies must be CUDA tensors."
    assert cos.device == hidden_states.device and sin.device == hidden_states.device, (
        "RoPE frequencies must be on the input device."
    )
    assert cos.dtype == hidden_states.dtype and sin.dtype == hidden_states.dtype, (
        "RoPE frequencies must match the input dtype."
    )
    assert cos.is_contiguous() and sin.is_contiguous(), (
        "RoPE frequencies must be contiguous."
    )
    assert cos.shape == (sequence_length, head_dim), (
        "RoPE cosine frequencies must have shape "
        f"({sequence_length}, {head_dim})."
    )
    assert sin.shape == (sequence_length, head_dim), (
        "RoPE sine frequencies must have shape "
        f"({sequence_length}, {head_dim})."
    )
    assert head_dim % 2 == 0, "RoPE head dimension must be even."
    assert head_dim == ROPE_BLOCK_SIZE, (
        f"RoPE head dimension must equal {ROPE_BLOCK_SIZE}; received {head_dim}."
    )

    output = torch.empty_like(hidden_states)
    _rope_kernel[(batch_size * num_heads * sequence_length,)](
        hidden_states,
        cos,
        sin,
        output,
        sequence_length=sequence_length,
        head_dim=head_dim,
        BLOCK_SIZE=ROPE_BLOCK_SIZE,
        num_warps=4,
    )
    return output


class Apertus1p5TextRotaryEmbedding(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        inverse_frequencies = 1.0 / (
            ROPE_THETA
            ** (
                torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM
            )
        )
        wavelengths = 2 * math.pi / inverse_frequencies
        low_frequency_wavelength = (
            ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS / ROPE_LOW_FREQ_FACTOR
        )
        high_frequency_wavelength = (
            ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS / ROPE_HIGH_FREQ_FACTOR
        )
        scaled_frequencies = torch.where(
            wavelengths > low_frequency_wavelength,
            inverse_frequencies / ROPE_FACTOR,
            inverse_frequencies,
        )
        smooth_factor = (
            ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS / wavelengths
            - ROPE_LOW_FREQ_FACTOR
        ) / (ROPE_HIGH_FREQ_FACTOR - ROPE_LOW_FREQ_FACTOR)
        smoothed_frequencies = (
            (1 - smooth_factor) * scaled_frequencies / ROPE_FACTOR
            + smooth_factor * scaled_frequencies
        )
        is_medium_frequency = (
            (wavelengths >= high_frequency_wavelength)
            & (wavelengths <= low_frequency_wavelength)
        )
        self.register_buffer(
            "inverse_frequencies",
            torch.where(is_medium_frequency, smoothed_frequencies, scaled_frequencies),
            persistent=False,
        )

    def forward(self, hidden_states, position_ids):
        frequencies = torch.outer(
            position_ids.float(), self.inverse_frequencies.float()
        )
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(hidden_states.dtype), angles.sin().to(hidden_states.dtype)




@triton.jit
def _causal_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    sequence_length: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_block = tl.program_id(axis=0)
    batch_head = tl.program_id(axis=1)
    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    feature_offsets = tl.arange(0, head_dim)
    batch_head_offset = batch_head * sequence_length * head_dim
    query_mask = query_offsets < sequence_length
    query_ptrs = (
        query_ptr
        + batch_head_offset
        + query_offsets[:, None] * head_dim
        + feature_offsets[None, :]
    )
    queries = tl.load(query_ptrs, mask=query_mask[:, None], other=0.0)
    max_scores = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    score_sums = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, head_dim), tl.float32)

    for key_block_start in range(0, sequence_length, BLOCK_N):
        current_key_offsets = key_block_start + key_offsets
        key_mask = current_key_offsets < sequence_length
        key_ptrs = (
            key_ptr
            + batch_head_offset
            + current_key_offsets[:, None] * head_dim
            + feature_offsets[None, :]
        )
        keys = tl.load(key_ptrs, mask=key_mask[:, None], other=0.0)
        scores = tl.dot(queries, tl.trans(keys)) * scale
        causal_mask = query_offsets[:, None] >= current_key_offsets[None, :]
        scores = tl.where(causal_mask, scores, -float("inf"))
        block_max_scores = tl.max(scores, axis=1)
        next_max_scores = tl.maximum(max_scores, block_max_scores)
        probabilities = tl.exp(scores - next_max_scores[:, None])
        rescale = tl.exp(max_scores - next_max_scores)
        score_sums = score_sums * rescale + tl.sum(probabilities, axis=1)
        value_ptrs = (
            value_ptr
            + batch_head_offset
            + current_key_offsets[:, None] * head_dim
            + feature_offsets[None, :]
        )
        values = tl.load(value_ptrs, mask=key_mask[:, None], other=0.0)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values
        )
        max_scores = next_max_scores

    output_ptrs = (
        output_ptr
        + batch_head_offset
        + query_offsets[:, None] * head_dim
        + feature_offsets[None, :]
    )
    tl.store(
        output_ptrs,
        accumulator / score_sums[:, None],
        mask=query_mask[:, None],
    )


def triton_causal_attention(query_states, key_states, value_states):
    batch_size, num_heads, sequence_length, head_dim = query_states.shape
    assert query_states.is_cuda, "Attention requires CUDA queries."
    assert key_states.is_cuda and value_states.is_cuda, (
        "Attention requires CUDA keys and values."
    )
    assert query_states.device == key_states.device == value_states.device, (
        "Attention inputs must be on the same device."
    )
    assert query_states.dtype in (torch.float16, torch.bfloat16), (
        "Attention queries must use float16 or bfloat16."
    )
    assert key_states.dtype == query_states.dtype == value_states.dtype, (
        "Attention inputs must use the same dtype."
    )
    assert query_states.is_contiguous(), "Attention queries must be contiguous."
    assert key_states.is_contiguous(), "Attention keys must be contiguous."
    assert value_states.is_contiguous(), "Attention values must be contiguous."
    assert key_states.shape == query_states.shape, (
        "Attention keys must have the same shape as queries."
    )
    assert value_states.shape == query_states.shape, (
        "Attention values must have the same shape as queries."
    )
    assert sequence_length > 0, "Attention sequence length must be positive."
    assert head_dim == ROPE_BLOCK_SIZE, (
        f"Attention head dimension must equal {ROPE_BLOCK_SIZE}; received {head_dim}."
    )
    assert not torch.is_grad_enabled(), "Attention does not support autograd."

    output = torch.empty_like(query_states)
    _causal_attention_kernel[
        (triton.cdiv(sequence_length, ATTENTION_BLOCK_M), batch_size * num_heads)
    ](
        query_states,
        key_states,
        value_states,
        output,
        sequence_length=sequence_length,
        head_dim=head_dim,
        scale=head_dim**-0.5,
        BLOCK_M=ATTENTION_BLOCK_M,
        BLOCK_N=ATTENTION_BLOCK_N,
        num_warps=4,
    )
    return output


class Apertus1p5TextAttention(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.q_proj = torch.nn.Linear(
            HIDDEN_SIZE, HIDDEN_SIZE, bias=False, device=device, dtype=dtype
        )
        self.k_proj = torch.nn.Linear(
            HIDDEN_SIZE,
            NUM_KEY_VALUE_HEADS * HEAD_DIM,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.v_proj = torch.nn.Linear(
            HIDDEN_SIZE,
            NUM_KEY_VALUE_HEADS * HEAD_DIM,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.o_proj = torch.nn.Linear(
            HIDDEN_SIZE, HIDDEN_SIZE, bias=False, device=device, dtype=dtype
        )
        self.q_norm = Apertus1p5TextRMSNorm(HEAD_DIM, device=device, dtype=dtype)
        self.k_norm = Apertus1p5TextRMSNorm(HEAD_DIM, device=device, dtype=dtype)

    def forward(self, hidden_states, cos, sin):
        batch_size, sequence_length, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            batch_size, sequence_length, NUM_ATTENTION_HEADS, HEAD_DIM
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, sequence_length, NUM_KEY_VALUE_HEADS, HEAD_DIM
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, sequence_length, NUM_KEY_VALUE_HEADS, HEAD_DIM
        ).transpose(1, 2)
        query_states = apply_rotary_embedding(self.q_norm(query_states), cos, sin)
        key_states = apply_rotary_embedding(self.k_norm(key_states), cos, sin)
        repeat_factor = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
        key_states = key_states.repeat_interleave(repeat_factor, dim=1)
        value_states = value_states.repeat_interleave(repeat_factor, dim=1)
        attention_output = triton_causal_attention(
            query_states,
            key_states,
            value_states,
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, sequence_length, HIDDEN_SIZE
        )
        return self.o_proj(attention_output)


class Apertus1p5TextMLP(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.up_proj = torch.nn.Linear(
            HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False, device=device, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(
            INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False, device=device, dtype=dtype
        )
        self.act_fn = XIELUActivation(device=device, dtype=dtype)

    def forward(self, hidden_states):
        hidden_states = triton_linear(hidden_states, self.up_proj.weight)
        hidden_states = self.act_fn(hidden_states)
        return triton_linear(hidden_states, self.down_proj.weight)


class Apertus1p5TextDecoderLayer(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.self_attn = Apertus1p5TextAttention(device, dtype)
        self.mlp = Apertus1p5TextMLP(device, dtype)
        self.attention_layernorm = Apertus1p5TextRMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype
        )
        self.feedforward_layernorm = Apertus1p5TextRMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype
        )

    def forward(self, hidden_states, cos, sin):
        hidden_states = hidden_states + self.self_attn(
            self.attention_layernorm(hidden_states), cos, sin
        )
        return hidden_states + self.mlp(self.feedforward_layernorm(hidden_states))


class Apertus1p5TextModel(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            padding_idx=3,
            device=device,
            dtype=dtype,
        )
        self.layers = torch.nn.ModuleList(
            Apertus1p5TextDecoderLayer(device, dtype) for _ in range(NUM_LAYERS)
        )
        self.norm = Apertus1p5TextRMSNorm(HIDDEN_SIZE, device=device, dtype=dtype)
        self.rotary_emb = Apertus1p5TextRotaryEmbedding(device=device)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        return self.norm(hidden_states)


class Apertus1p5TextForCausalLM(torch.nn.Module):
    def __init__(self, device, dtype=torch.bfloat16):
        super().__init__()
        self.model = Apertus1p5TextModel(device, dtype)
        self.lm_head = torch.nn.Linear(
            HIDDEN_SIZE,
            OUTPUT_VOCAB_SIZE,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=32):
        end_token_ids = torch.tensor(EOS_TOKEN_IDS, device=input_ids.device)
        for _ in range(max_new_tokens):
            next_token = self(input_ids)[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if torch.isin(next_token, end_token_ids).all():
                break
        return input_ids


def load_pretrained_text_model(device):
    model = Apertus1p5TextForCausalLM("meta")
    model.to_empty(device=device)
    model.model.rotary_emb = Apertus1p5TextRotaryEmbedding(device=device)
    index_path = MODEL_WEIGHTS_DIR / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    model_state = model.state_dict()
    loaded_keys = 0

    with torch.no_grad():
        with ExitStack() as stack:
            checkpoints = {
                shard: stack.enter_context(
                    safe_open(MODEL_WEIGHTS_DIR / shard, framework="pt", device="cpu")
                )
                for shard in set(weight_map.values())
            }
            for model_key, parameter in model_state.items():
                checkpoint_key = "model.language_model." + model_key.removeprefix(
                    "model."
                )
                if model_key == "lm_head.weight":
                    checkpoint_key = model_key
                checkpoint_shard = weight_map.get(checkpoint_key)
                if checkpoint_shard is None:
                    raise KeyError(f"Missing checkpoint tensor: {checkpoint_key}")
                tensor = checkpoints[checkpoint_shard].get_tensor(checkpoint_key)
                if tensor.shape != parameter.shape:
                    raise ValueError(
                        f"Shape mismatch for {checkpoint_key}: {tensor.shape} != "
                        f"{parameter.shape}"
                    )
                parameter.copy_(tensor.to(device=device, dtype=parameter.dtype))
                loaded_keys += 1

    print(f"Loaded {loaded_keys} text-model tensors from the checkpoint.")
    return model.eval()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Text inference requires a CUDA GPU for this model.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    model = load_pretrained_text_model("cuda")
    messages = [
        {"role": "system", "content": "You are a concise and helpful assistant."},
        {"role": "user", "content": "You are a good swiss citizen, Who has the best cheese in the world?"}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to("cuda")
    torch.cuda.synchronize()
    generation_start = time.perf_counter()
    output_ids = model.generate(input_ids, max_new_tokens=640)
    torch.cuda.synchronize()
    generated_tokens = output_ids.shape[-1] - input_ids.shape[-1]
    generation_seconds = time.perf_counter() - generation_start
    print(f"Generation throughput: {generated_tokens / generation_seconds:.2f} tokens/s")
    print(tokenizer.decode(output_ids[0, input_ids.shape[-1] :], skip_special_tokens=True))
