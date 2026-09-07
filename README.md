# Hugging Face GGUF Quantizer

An interactive terminal application for downloading Hugging Face language models, converting them to **GGUF**, and generating quantized models using **llama.cpp**.

No command-line arguments required.

Simply run:

```bash
python quantize.py
```

and use the interactive menu.

---

## What it does

The application performs this pipeline:

```text
Hugging Face Model
        │
        ▼
Download / reuse checkpoint
        │
        ▼
BF16 GGUF
        │
        ├─────────────► Q8_0
        ├─────────────► Q6_K
        ├─────────────► Q5_K_M
        ├─────────────► Q4_K_M
        ├─────────────► Q3_K_M
        ├─────────────► Q2_K
        ├─────────────► IQ2_M
        └─────────────► IQ1_M
```

The Hugging Face checkpoint is downloaded only once and can be reused.

The BF16 GGUF can also be reused to generate additional quantizations without repeating the model conversion.

---

# Features

* Interactive CLI
* No command-line arguments
* Accepts Hugging Face repository IDs
* Accepts local Hugging Face model directories
* Automatic Hugging Face download
* Supports gated/private models using `HF_TOKEN`
* Automatically clones `llama.cpp` if missing
* Automatically builds `llama.cpp` if required
* Converts Hugging Face checkpoints to GGUF
* Generates one or many quantization formats
* Supports Q8 through extreme IQ1 variants
* Reuses existing downloads
* Reuses existing BF16 GGUF models
* Displays final file sizes
* Handles unsupported quantizations individually instead of terminating the entire run
* Clean terminal UI using Rich
* Works without an NVIDIA GPU

---

# Requirements

You need:

* Python 3.10+
* Git
* CMake
* A C/C++ compiler
* Sufficient RAM
* Sufficient disk space

A CUDA GPU is **not required**.

Quantization can run on the CPU.

On Apple Silicon systems, llama.cpp can also use Apple's Metal backend for inference.

---

# Installation

Clone or create your project:

```bash
mkdir hf-quantizer
cd hf-quantizer
```

Place these files inside:

```text
hf-quantizer/
├── quantize.py
├── requirements.txt
└── README.md
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate it.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running

Start the application:

```bash
python quantize.py
```

The first prompt asks for a model:

```text
╭─────────────────────────────────────────────────────────╮
│                HF MODEL QUANTIZER                       │
│       Hugging Face → GGUF → llama.cpp quantization     │
╰─────────────────────────────────────────────────────────╯

Hugging Face model ID or local HF directory:
```

For example:

```text
google/gemma-3-4b-it
```

or:

```text
meta-llama/Llama-3.2-3B-Instruct
```

or a local model:

```text
/Users/me/models/my-model
```

---

# Quantization menu

You will then see something similar to:

```text
╭────┬─────────┬───────────────┬────────────────────────────────────╮
│ #  │ Format  │ Class         │ Description                        │
├────┼─────────┼───────────────┼────────────────────────────────────┤
│ 1  │ Q8_0    │ Standard      │ Very high quality                  │
│ 2  │ Q6_K    │ Standard      │ Excellent quality                  │
│ 3  │ Q5_K_M  │ Standard      │ High quality                       │
│ 4  │ Q4_K_M  │ Standard      │ Recommended general-purpose Q4    │
│ 5  │ Q4_K_S  │ Standard      │ Smaller Q4                         │
│ 6  │ Q4_0    │ Standard      │ Classic Q4                         │
│ 7  │ Q3_K_M  │ Low-bit       │ Aggressive compression            │
│ 8  │ Q3_K_S  │ Low-bit       │ Smaller Q3                         │
│ 9  │ Q2_K    │ Ultra-low-bit │ Very aggressive compression       │
│ 10 │ IQ4_XS  │ IQ            │ Importance-aware Q4               │
│ 11 │ IQ3_M   │ IQ            │ Importance-aware Q3               │
│ 12 │ IQ2_M   │ IQ            │ Importance-aware low-bit          │
│ 13 │ IQ2_XS  │ IQ            │ Smaller IQ2                       │
│ 14 │ IQ2_XXS │ Experimental  │ Extremely aggressive              │
│ 15 │ IQ1_M   │ Experimental  │ Extreme compression               │
│ 16 │ IQ1_S   │ Experimental  │ Extreme compression               │
╰────┴─────────┴───────────────┴────────────────────────────────────╯
```

Choose one:

```text
4
```

to generate:

```text
Q4_K_M
```

---

# Generate multiple quantizations

You can select several formats at once.

For example:

```text
4,7,9,12,15
```

generates:

```text
Q4_K_M
Q3_K_M
Q2_K
IQ2_M
IQ1_M
```

This is particularly useful for experiments comparing how model behavior changes as quantization becomes increasingly aggressive.

---

# Generate everything

Enter:

```text
all
```

The application will attempt every configured quantization.

Unsupported formats are skipped without terminating the full experiment.

---

# Output structure

Downloaded Hugging Face checkpoints are stored under:

```text
models/hf/
```

Generated GGUF models are stored under:

```text
models/gguf/
```

Example:

```text
models/
├── hf/
│   └── gemma-3-4b-it/
│
└── gguf/
    └── gemma-3-4b-it/
        ├── gemma-3-4b-it-BF16.gguf
        ├── gemma-3-4b-it-Q8_0.gguf
        ├── gemma-3-4b-it-Q6_K.gguf
        ├── gemma-3-4b-it-Q5_K_M.gguf
        ├── gemma-3-4b-it-Q4_K_M.gguf
        ├── gemma-3-4b-it-Q3_K_M.gguf
        ├── gemma-3-4b-it-Q2_K.gguf
        ├── gemma-3-4b-it-IQ2_M.gguf
        └── gemma-3-4b-it-IQ1_M.gguf
```

---

# Private and gated Hugging Face models

Some models require authentication.

For example, certain Gemma or Llama repositories may require accepting their license on Hugging Face first.

Set your Hugging Face token:

### macOS / Linux

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

### Windows PowerShell

```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
```

Then run normally:

```bash
python quantize.py
```

The program automatically reads `HF_TOKEN`.

---

# llama.cpp

You do not have to manually install llama.cpp.

If this directory is missing:

```text
./llama.cpp
```

the application asks:

```text
llama.cpp was not found.
Clone llama.cpp automatically? [Y/n]
```

Choosing Yes runs:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
```

If `llama-quantize` has not been compiled, the application offers to build it using CMake.

---

# Manual llama.cpp installation

You can also install it yourself:

```bash
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j

cd ..
```

Then run:

```bash
python quantize.py
```

---

# How quantization works

The application intentionally uses two stages.

First:

```text
Hugging Face checkpoint
        │
        ▼
BF16 GGUF
```

Then:

```text
BF16 GGUF
        │
        ▼
llama-quantize
        │
        ▼
Quantized GGUF
```

This makes the BF16 model a reusable source.

For example, after converting the model once:

```text
                   ┌── Q8
                   ├── Q6
                   ├── Q5
BF16 GGUF ─────────┼── Q4
                   ├── Q3
                   ├── Q2
                   ├── IQ2
                   └── IQ1
```

You do not need to download or convert the original Hugging Face model again for every quantization.

---

# Q4 does not mean exactly four bits per model parameter

Names such as:

```text
Q4
Q3
Q2
IQ1
```

describe quantization families.

They should not be interpreted as exact whole-model bits-per-parameter values.

Quantized formats also require information such as:

```text
weights
scales
block metadata
quantization parameters
tensor metadata
```

Therefore:

```text
Q4_K_M ≠ exactly 4.000 bits per parameter
```

and:

```text
IQ1_M ≠ exactly 1.000 bit per parameter
```

The actual effective bits-per-weight depend on the format and architecture.

---

# How low can we go?

For conventional binary representation, storing each independent weight with only two possible states:

```text
-1
+1
```

requires one bit of state information:

```text
0 → -1
1 → +1
```

This gives binary neural-network weights.

However, achieving an **average model storage below one bit per parameter** is conceptually possible through techniques such as:

```text
sparsity
entropy coding
shared codebooks
run-length encoding
structured weights
vector quantization
weight prediction
procedural reconstruction
```

That is no longer simple fixed-width scalar quantization.

So a theoretical experimental pipeline could look like:

```text
FP32
 │
FP16
 │
Q8
 │
Q6
 │
Q5
 │
Q4
 │
Q3
 │
Q2
 │
binary weights
 │
structured binary weights
 │
compressed representation < 1 average bit/parameter
```

---

# Important: IQ1 is not literally one bit per weight

The `IQ1_*` name can be misleading.

The quantized weight representation itself is extremely low-bit, but the model also needs scales and other block information.

Therefore actual model storage is higher than exactly one bit per parameter.

Treat IQ1 as an **extreme low-bit quantization family**, rather than as a literal one-bit file representation.

---

# Recommended experiments

For examining quantization degradation, a useful sequence is:

```text
BF16
 ↓
Q8_0
 ↓
Q6_K
 ↓
Q5_K_M
 ↓
Q4_K_M
 ↓
Q3_K_M
 ↓
Q2_K
 ↓
IQ2_M
 ↓
IQ1_M
```

Then give every model exactly the same prompts and compare:

```text
perplexity
generation quality
reasoning
instruction following
repetition
grammar
coherence
token generation speed
RAM usage
model size
```

This gives a much better picture of **where model capability begins collapsing** than simply comparing file sizes.

---

# Example experiment

Run:

```bash
python quantize.py
```

Model:

```text
google/gemma-3-4b-it
```

Select:

```text
1,4,7,9,12,15
```

You will get:

```text
Gemma BF16
Gemma Q8_0
Gemma Q4_K_M
Gemma Q3_K_M
Gemma Q2_K
Gemma IQ2_M
Gemma IQ1_M
```

You can then load every GGUF with exactly the same inference settings and compare behavior.

---

# Disk and memory warning

Quantizing large models requires significant disk space.

During conversion you may temporarily have:

```text
Original HF checkpoint
+
BF16 GGUF
+
one or more quantized GGUF files
```

For large models this can consume substantial storage.

You also need sufficient RAM for the conversion/quantization process.

A GPU is not mandatory.

---

# Model compatibility

Not every Hugging Face architecture can necessarily be converted.

Compatibility depends on whether the current version of llama.cpp understands that architecture and its tensor layout.

If conversion fails with an unsupported architecture error, updating llama.cpp is often the first thing to try:

```bash
cd llama.cpp

git pull

cmake --build build --config Release -j

cd ..
```

Then run:

```bash
python quantize.py
```

again.

---

# Updating llama.cpp

Because model support and quantization formats evolve quickly, periodically update it:

```bash
cd llama.cpp

git pull

cmake --build build --config Release -j

cd ..
```

---

# Project structure

After some experiments your project may look like:

```text
hf-quantizer/
│
├── quantize.py
├── requirements.txt
├── README.md
│
├── llama.cpp/
│
└── models/
    ├── hf/
    │   ├── gemma-3-4b-it/
    │   └── another-model/
    │
    └── gguf/
        ├── gemma-3-4b-it/
        │   ├── gemma-3-4b-it-BF16.gguf
        │   ├── gemma-3-4b-it-Q4_K_M.gguf
        │   └── ...
        │
        └── another-model/
```

---

# Goal

The project is designed not only to create smaller models, but also to make quantization experiments easy:

```text
How much numerical information
can we remove from an LLM
before its learned behavior disappears?
```

That makes it useful for comparing the relationship between:

```text
parameter precision
model size
language quality
reasoning
stability
and emergent behavior
```
