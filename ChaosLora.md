## 🌀 Chaos LoRA

Chaos LoRA is an experimental tool for generating **deterministic, untrained LoRA adapters** for Hugging Face language models.

Traditional LoRA learns low-rank matrices from a training dataset.

Chaos LoRA does something different:

```text
Training data       ❌
Backpropagation     ❌
Optimizer            ❌
Fine-tuning          ❌

Mathematical rule    ✅
Deterministic seed   ✅
Low-rank update      ✅
PEFT adapter          ✅
```

Instead of training the LoRA matrices `A` and `B`, Chaos LoRA generates them directly from mathematical functions.

The resulting model update is:

```text
ΔW = strength × (B @ A)

W' = W + ΔW
```

This makes it possible to study how structured mathematical perturbations affect the behavior of pretrained language models.

### Why?

The goal is not to improve the model.

The goal is to experiment with questions such as:

- How sensitive are language models to structured low-rank perturbations?
- At what perturbation strength does model behavior begin to change?
- Do different mathematical sequences produce different behavioral effects?
- How does deterministic chaos compare with ordinary random noise?
- Which parts of a transformer are most sensitive to perturbation?
- Can coherent behavior survive surprisingly large structured perturbations?

Because the adapters are deterministic, experiments can be reproduced exactly using the same:

```text
model
function
rank
seed
strength
target layers
```

---

### Supported generators

| Generator | Type | Description |
|---|---|---|
| `golden` | Quasiperiodic | Sequence based on the golden ratio φ |
| `sine` | Periodic | Deterministic sine sequence |
| `cosine` | Periodic | Deterministic cosine sequence |
| `pi` | Periodic | π-frequency sine sequence |
| `gaussian` | Random-like | Seeded Gaussian distribution |
| `uniform` | Random-like | Seeded uniform distribution |
| `logistic` | Chaotic | Logistic map with `r = 4` |
| `tent` | Chaotic | Deterministic tent map |

The same seed always produces the same adapter.

---

### Target layers

Chaos LoRA automatically inspects the Hugging Face model and discovers its `torch.nn.Linear` layers.

You can perturb:

```text
1. All Linear layers
2. Attention layers
3. MLP / FFN layers
4. Layers matching a custom regex
```

This avoids hardcoding architectures such as Llama, Gemma, Qwen, or Muse Glimmer.

For example:

```text
Hugging Face model: Qwen/Qwen2.5-0.5B-Instruct

Detected 145 Linear layers.

Target:

1  All Linear layers
2  Attention layers
3  MLP / FFN layers
4  Custom regex
```

---

### Generate an adapter

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python chaos_lora.py
```

Example configuration:

```text
Hugging Face model: Qwen/Qwen2.5-0.5B-Instruct

Function:
golden

Rank:
8

Seed:
42

Strength:
0.01

Target:
all
```

The adapter is written to:

```text
chaos_adapters/
└── Qwen2.5-0.5B-Instruct/
    └── golden-r8-seed42-s0.01-all/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── chaos_config.json
```

`chaos_config.json` records the experiment parameters so the perturbation can be reproduced later.

---

### Load the Chaos LoRA

Chaos LoRA produces a standard PEFT-style adapter.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

adapter = (
    "chaos_adapters/"
    "Qwen2.5-0.5B-Instruct/"
    "golden-r8-seed42-s0.01-all"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    model,
    adapter,
)

prompt = "Explain why the sky is blue."

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False,
)

print(
    tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )
)
```

---

### Strength

`strength` directly controls the magnitude of the low-rank perturbation:

```text
ΔW = strength × BA
```

For example:

```text
0       Original model
0.001   Very small perturbation
0.01    Small perturbation
0.1     Moderate experiment
1.0     Full generated perturbation
>1      Extreme perturbation
```

These values are experimental rather than universal thresholds. Different models, ranks, generators, and target layers can respond very differently.

A useful sweep is:

```text
0
0.0001
0.001
0.01
0.03
0.1
0.3
1.0
3.0
```

---

### Reproducibility

Chaos LoRA is deterministic.

For a fixed:

```text
model = Qwen/Qwen2.5-0.5B-Instruct
function = golden
rank = 8
seed = 42
strength = 0.01
targets = all
```

the generated mathematical matrices are reproducible.

Changing only the seed produces a different deterministic perturbation.

This allows experiments such as:

```text
Golden seed 42
vs
Golden seed 43
vs
Gaussian seed 42
vs
Logistic seed 42
```

while keeping the model, rank, strength, prompt, and inference settings constant.

---

### Suggested experiment

Start with a small model.

Generate:

```text
Original

Golden:
  0.001
  0.01
  0.1
  1.0

Gaussian:
  0.001
  0.01
  0.1
  1.0

Logistic:
  0.001
  0.01
  0.1
  1.0
```

Use identical prompts and deterministic generation settings.

Then compare:

```text
Output divergence
Semantic similarity
Perplexity
Token probability changes
Benchmark accuracy
Coherence
Repetition
Failure point
```

This separates the effect of the mathematical perturbation from sampling randomness.

---

### Research status

Chaos LoRA is experimental.

It is intended for exploring deterministic perturbations of neural networks, not as a model-improvement or fine-tuning technique.

The central experiment is:

> What happens when the learned weights of a language model are influenced by deterministic low-rank mathematical structures that were never learned from data?