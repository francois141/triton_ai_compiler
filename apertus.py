import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as functional
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


class Apertus1p5TextRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=RMS_NORM_EPS, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


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
        alpha_p = functional.softplus(self.alpha_p)
        alpha_n = self.beta + functional.softplus(self.alpha_n)
        return torch.where(
            hidden_states > 0,
            alpha_p * hidden_states * hidden_states + self.beta * hidden_states,
            (torch.expm1(torch.min(hidden_states, self.eps)) - hidden_states)
            * alpha_n
            + self.beta * hidden_states,
        )


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


def apply_rotary_embedding(hidden_states, cos, sin):
    rotated = torch.cat(
        (
            -hidden_states[..., hidden_states.shape[-1] // 2 :],
            hidden_states[..., : hidden_states.shape[-1] // 2],
        ),
        dim=-1,
    )
    return hidden_states * cos[None, None] + rotated * sin[None, None]


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
        attention_output = functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
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
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


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
        {"role": "user", "content": "Who has the best cheese in the world?"}
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
    output_ids = model.generate(input_ids, max_new_tokens=64)
    print(tokenizer.decode(output_ids[0, input_ids.shape[-1] :], skip_special_tokens=True))
