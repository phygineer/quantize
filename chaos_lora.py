#!/usr/bin/env python3

"""
Generic Chaos LoRA Generator
============================

Creates deterministic, untrained LoRA adapters for Hugging Face models.

Instead of learning LoRA matrices A and B through gradient descent, this tool
constructs them from deterministic mathematical functions.

    ΔW = strength * (B @ A)
    W' = W + ΔW

Supported generators:
    - golden
    - sine
    - cosine
    - pi
    - gaussian
    - uniform
    - logistic
    - tent

The generated adapter is saved in standard PEFT-compatible format.

No training dataset is required.
No optimizer is required.
No GGUF/llama.cpp conversion is required for Hugging Face inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.prompt import FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


console = Console()

OUTPUT_ROOT = Path("chaos_adapters")
CACHE_ROOT = Path("models/hf")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChaosConfig:
    model_id: str
    function: str
    rank: int
    seed: int
    strength: float
    target_mode: str
    target_regex: str | None = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def stable_seed(name: str, seed: int, suffix: str = "") -> int:
    """
    Produce a stable per-layer seed.

    Python's built-in hash() is intentionally randomized between processes,
    so SHA256 is used instead.
    """
    text = f"{seed}:{name}:{suffix}".encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def rms_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x.float() ** 2) + eps)
    return x / rms


def safe_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------------
# Mathematical generators
# ---------------------------------------------------------------------------

def indexed_positions(numel: int) -> torch.Tensor:
    return torch.arange(numel, dtype=torch.float64)


def generate_sine(numel: int, seed: int) -> torch.Tensor:
    x = indexed_positions(numel)
    phase = (seed % 100000) / 1000.0
    return torch.sin(x + phase).float()


def generate_cosine(numel: int, seed: int) -> torch.Tensor:
    x = indexed_positions(numel)
    phase = (seed % 100000) / 1000.0
    return torch.cos(x + phase).float()


def generate_golden(numel: int, seed: int) -> torch.Tensor:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    x = indexed_positions(numel)
    phase = (seed % 100000) / 1000.0
    return torch.sin(x * phi + phase).float()


def generate_pi(numel: int, seed: int) -> torch.Tensor:
    x = indexed_positions(numel)
    phase = (seed % 100000) / 1000.0
    return torch.sin(x * math.pi + phase).float()


def generate_gaussian(numel: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(numel, generator=generator)


def generate_uniform(numel: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(numel, generator=generator) * 2.0 - 1.0


def initial_chaos_state(seed: int) -> float:
    """
    Deterministically map an integer seed into (0, 1), avoiding exact
    endpoints and common pathological starting values.
    """
    digest = hashlib.sha256(str(seed).encode()).digest()
    integer = int.from_bytes(digest[:8], "little")

    x = (integer + 1) / (2**64 + 2)

    # Avoid values extremely close to 0, 0.5 or 1.
    x = 0.001 + x * 0.998

    if abs(x - 0.5) < 1e-7:
        x += 1e-5

    return x


def generate_logistic(numel: int, seed: int) -> torch.Tensor:
    """
    Logistic map:

        x[n+1] = 4 * x[n] * (1 - x[n])

    Values are remapped from [0,1] to [-1,1].
    """
    x = initial_chaos_state(seed)

    # Burn-in removes some dependence on the arbitrary initial transient.
    for _ in range(100):
        x = 4.0 * x * (1.0 - x)

    values = torch.empty(numel, dtype=torch.float32)

    for i in range(numel):
        x = 4.0 * x * (1.0 - x)
        values[i] = 2.0 * x - 1.0

    return values


def generate_tent(numel: int, seed: int) -> torch.Tensor:
    """
    Tent map:

        x[n+1] = 2x[n]              if x < 0.5
                 2(1 - x[n])        otherwise

    Values are remapped to [-1,1].
    """
    x = initial_chaos_state(seed)

    values = torch.empty(numel, dtype=torch.float32)

    for i in range(numel):
        if x < 0.5:
            x = 2.0 * x
        else:
            x = 2.0 * (1.0 - x)

        values[i] = 2.0 * x - 1.0

    return values


GENERATORS: dict[str, Callable[[int, int], torch.Tensor]] = {
    "golden": generate_golden,
    "sine": generate_sine,
    "cosine": generate_cosine,
    "pi": generate_pi,
    "gaussian": generate_gaussian,
    "uniform": generate_uniform,
    "logistic": generate_logistic,
    "tent": generate_tent,
}


def generate_matrix(
    shape: tuple[int, ...],
    function: str,
    seed: int,
) -> torch.Tensor:

    numel = math.prod(shape)

    values = GENERATORS[function](numel, seed)
    values = rms_normalize(values)

    return values.reshape(shape)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    """
    Load model on CPU.

    We only need the architecture and Linear layer dimensions.
    """
    console.print(f"\n[cyan]Loading architecture:[/] {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    return model


def discover_linear_layers(model) -> dict[str, nn.Linear]:
    layers = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers[name] = module

    return layers


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

ATTENTION_HINTS = (
    "attn",
    "attention",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "query",
    "key",
    "value",
)

MLP_HINTS = (
    "mlp",
    "ffn",
    "feed_forward",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
)


def contains_hint(name: str, hints: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in hints)


def select_layers(
    layers: dict[str, nn.Linear],
    mode: str,
    regex: str | None,
) -> dict[str, nn.Linear]:

    if mode == "all":
        return layers

    if mode == "attention":
        return {
            name: layer
            for name, layer in layers.items()
            if contains_hint(name, ATTENTION_HINTS)
        }

    if mode == "mlp":
        return {
            name: layer
            for name, layer in layers.items()
            if contains_hint(name, MLP_HINTS)
        }

    if mode == "regex":
        if not regex:
            raise ValueError("Regex mode requires a regular expression.")

        pattern = re.compile(regex)

        return {
            name: layer
            for name, layer in layers.items()
            if pattern.search(name)
        }

    raise ValueError(f"Unknown target mode: {mode}")


# ---------------------------------------------------------------------------
# Adapter generation
# ---------------------------------------------------------------------------

def generate_adapter(
    config: ChaosConfig,
    layers: dict[str, nn.Linear],
    output_dir: Path,
):
    tensors = {}

    table = Table(title="Chaos LoRA Layers")
    table.add_column("Layer")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Rank", justify="right")

    # Linear strength:
    #
    # ΔW = strength * B @ A
    #
    # We distribute sqrt(strength) across A and B.
    magnitude = math.sqrt(abs(config.strength))

    # Preserve negative strengths too.
    sign = -1.0 if config.strength < 0 else 1.0

    for name, module in layers.items():

        in_features = module.in_features
        out_features = module.out_features

        rank = min(config.rank, in_features, out_features)

        seed_a = stable_seed(name, config.seed, "A")
        seed_b = stable_seed(name, config.seed, "B")

        A = generate_matrix(
            (rank, in_features),
            config.function,
            seed_a,
        )

        B = generate_matrix(
            (out_features, rank),
            config.function,
            seed_b,
        )

        # We want:
        #
        # B_scaled @ A_scaled
        # =
        # strength * (B @ A)
        #
        A *= magnitude
        B *= magnitude * sign

        A = A.to(torch.float16).contiguous()
        B = B.to(torch.float16).contiguous()

        key_a = f"base_model.model.{name}.lora_A.weight"
        key_b = f"base_model.model.{name}.lora_B.weight"

        tensors[key_a] = A
        tensors[key_b] = B

        table.add_row(
            name,
            str(in_features),
            str(out_features),
            str(rank),
        )

    console.print(table)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_file(
        tensors,
        output_dir / "adapter_model.safetensors",
    )

    # target_modules can be explicit full module paths. This avoids assuming
    # architecture-specific names such as q_proj/down_proj.
    target_modules = list(layers.keys())

    adapter_config = {
        "base_model_name_or_path": config.model_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": False,
        "lora_alpha": config.rank,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": config.rank,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }

    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    chaos_metadata = {
        "model": config.model_id,
        "function": config.function,
        "rank": config.rank,
        "seed": config.seed,
        "strength": config.strength,
        "target_mode": config.target_mode,
        "target_regex": config.target_regex,
        "formula": "delta_W = strength * (B @ A)",
        "training": False,
        "layers": list(layers.keys()),
    }

    with open(output_dir / "chaos_config.json", "w") as f:
        json.dump(chaos_metadata, f, indent=2)

    return tensors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def choose_function() -> str:

    table = Table(title="Mathematical Generator")

    table.add_column("#")
    table.add_column("Function")
    table.add_column("Description")

    choices = [
        ("1", "golden", "Golden-ratio quasiperiodic pattern"),
        ("2", "sine", "Deterministic sine sequence"),
        ("3", "cosine", "Deterministic cosine sequence"),
        ("4", "pi", "π-frequency sine sequence"),
        ("5", "gaussian", "Seeded Gaussian noise"),
        ("6", "uniform", "Seeded uniform noise"),
        ("7", "logistic", "Chaotic logistic map"),
        ("8", "tent", "Chaotic tent map"),
    ]

    for row in choices:
        table.add_row(*row)

    console.print(table)

    value = Prompt.ask(
        "Choose function",
        choices=[x[0] for x in choices],
        default="1",
    )

    return choices[int(value) - 1][1]


def choose_target():
    console.print(
        Panel(
            """
[1] All Linear layers
[2] Attention layers
[3] MLP / FFN layers
[4] Custom regular expression
""",
            title="Target Layers",
        )
    )

    choice = Prompt.ask(
        "Target",
        choices=["1", "2", "3", "4"],
        default="1",
    )

    mapping = {
        "1": "all",
        "2": "attention",
        "3": "mlp",
        "4": "regex",
    }

    mode = mapping[choice]
    regex = None

    if mode == "regex":
        regex = Prompt.ask("Layer regex")

    return mode, regex


def main():

    console.print(
        Panel.fit(
            "[bold]Generic Chaos LoRA Generator[/]\n"
            "Deterministic mathematical perturbations without training",
            border_style="cyan",
        )
    )

    model_id = Prompt.ask(
        "Hugging Face model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )

    function = choose_function()

    rank = IntPrompt.ask(
        "LoRA rank",
        default=8,
    )

    seed = IntPrompt.ask(
        "Seed",
        default=42,
    )

    strength = FloatPrompt.ask(
        "Chaos strength",
        default=0.01,
    )

    target_mode, target_regex = choose_target()

    config = ChaosConfig(
        model_id=model_id,
        function=function,
        rank=rank,
        seed=seed,
        strength=strength,
        target_mode=target_mode,
        target_regex=target_regex,
    )

    model = load_model(model_id)

    all_layers = discover_linear_layers(model)

    console.print(
        f"\nDetected [bold]{len(all_layers)}[/] Linear layers."
    )

    selected_layers = select_layers(
        all_layers,
        target_mode,
        target_regex,
    )

    if not selected_layers:
        raise RuntimeError(
            "No Linear layers matched the selected target."
        )

    console.print(
        f"Selected [bold green]{len(selected_layers)}[/] layers."
    )

    strength_name = format(config.strength, "g")

    output_dir = (
        OUTPUT_ROOT
        / safe_name(model_id)
        / (
            f"{function}"
            f"-r{rank}"
            f"-seed{seed}"
            f"-s{strength_name}"
            f"-{target_mode}"
        )
    )

    generate_adapter(
        config,
        selected_layers,
        output_dir,
    )

    del model

    console.print(
        Panel.fit(
            f"""
[bold green]Chaos LoRA created![/]

Model:     {model_id}
Function:  {function}
Rank:      {rank}
Seed:      {seed}
Strength:  {strength}
Targets:   {target_mode}
Layers:    {len(selected_layers)}

Output:

{output_dir}

Formula:

ΔW = {strength} × (B @ A)
""",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()