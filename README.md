# 54.5M Small Language Model (SLM) from Scratch on Apple Silicon & CUDA

A decoder-only causal language model built entirely in pure PyTorch (no Hugging Face `transformers` model abstractions, no third-party transformer libraries) and trained from scratch on consumer hardware (Apple Silicon MPS and NVIDIA CUDA).

---

## 1. Model Architecture & Specifications

The current active model is a **54.5M parameter** modern transformer designed for high depth-to-width efficiency with Grouped-Query Attention (GQA) and SwiGLU:

| Hyperparameter | Value | Description |
|---|---|---|
| **Total Parameters** | **54,538,752** | Exact analytical & instantiated parameter count |
| **Non-Embedding Parameters** | **37,761,536** | Transformer block weights (69.2% of total) |
| **Embedding Parameters** | **16,777,216** | Tied input/output token embedding (30.8% of total) |
| **Layers ($N_{\text{layers}}$)** | **12** | Decoder transformer blocks |
| **Hidden Size ($d_{\text{model}}$)** | **512** | Model hidden dimension |
| **Query Heads** | **8** | Multi-head attention query projections |
| **KV Heads (GQA)** | **4** | Grouped-Query Attention (2 queries per KV group) |
| **Head Dimension** | **64** | $512 / 8 = 64$ |
| **MLP Intermediate Size** | **1,536** | SwiGLU gated linear unit ($3 \times$ projections) |
| **Vocabulary Size** | **32,768** | Byte-level BPE tokenizer (`tokenizer/tokenizer.json`) |
| **Max Sequence Length** | **1,024** | Context window |
| **Positional Embeddings** | **RoPE** | Rotary Position Embeddings ($\theta = 10,000$) |
| **Normalization** | **RMSNorm** | Pre-layer normalization ($\epsilon = 10^{-5}$) |
| **Weight Tying** | **Enabled** | LM Head shares weights with Token Embedding |

---

## 2. Training Progress & Milestones

### Pretraining: 236.98M Tokens (`exp_008`)
* **Tokens Trained**: **236,978,176 tokens** (~237M tokens total) across sequential non-regressive continuation stages (`exp_001` → `exp_005` → `exp_008`).
* **Step Count**: **7,232 steps** at 32,768 tokens/step (effective batch: 4 micro-batch $\times$ 8 gradient accumulation steps $\times$ 1,024 context).
* **Schedule**: Warmup-Stable-Decay (WSD) with high-quality quality annealing.
* **Checkpoint**: `experiments/exp_008/checkpoints/best.pt` (step 7,232, validation loss: **0.379**, val perplexity: **1.46** on the continuation slice).

### Supervised Fine-Tuning: 10M Tokens (`exp_009_sft`)
* **Dataset**: 10M tokens strictly balanced across 5 domains (45% SmolTalk, 20% Tulu 3 English, 15% Verified Math, 10% AST-parsed Coding, 10% Conversational Q&A).
* **Format**: Standard OpenAI / ChatML turn structure (`User:` / `Assistant:`).
* **Performance**: SFT validation loss **1.86**, perplexity **6.41**. Instruction-following and turn-taking format adherence achieved.

---

## 3. Dataset Pipeline & Historical Hash Registry

To ensure **100% fresh, non-overlapping documents** across training stages (30M, 60M, 100M, and 1B targets), the repository includes a precomputed document deduplication registry:

* **Hash Registry**: [`data/historical_doc_hashes.json.gz`](data/historical_doc_hashes.json.gz) contains **363,966 unique SHA-256 hashes** of every document previously seen across all pretraining and SFT datasets.
* **Automated Deduplication**: `load_all_past_hashes()` in dataset builders loads this compressed index in ~0.1s on any machine before streaming from Hugging Face, guaranteeing zero duplication without storing hundreds of megabytes of raw text in Git.
* **Document-Aware Masking**: Pretraining shards use uint16 binary token arrays with accompanying `_seg.bin` document boundary masks to prevent cross-document attention contamination in packed sequences.

---

## 4. Hardware & Cross-Platform Support

The training framework automatically detects available compute and optimizes precision:
* **Apple Silicon (Mac M1/M2/M3/M4)**: Uses Metal Performance Shaders (`mps`) with native `bfloat16`/`float16` autocasting. Measured throughput: ~1,640 tokens/sec.
* **NVIDIA GPUs (Linux/Windows PC)**: Uses `cuda` with automatic `bfloat16`/`float16` mixed precision and CUDA graph compatibility.
* **CPU**: Automatic fallback for debugging.

---

## 5. Repository Structure

```
SLM/
├── configs/
│   ├── model.yaml                    # 54.5M model architecture definition
│   ├── train.yaml                    # Base training hyperparameters & WSD schedule
│   ├── train_continuation.yaml       # Continuation training config (exp_008 baseline)
│   └── train_sft.yaml                # Supervised fine-tuning configuration
├── data/
│   ├── historical_doc_hashes.json.gz # 363,966 SHA-256 historical deduplication hashes
│   ├── manifest.json                 # Initial 60M pretraining manifest
│   └── metadata.json
├── data_100m/
│   └── manifest.json                 # 100M multi-domain continuation manifest
├── data_sft/
│   └── manifest.json                 # 10M balanced SFT mixture manifest
├── tokenizer/
│   ├── tokenizer.json                # Trained Byte-level BPE tokenizer (32,768 vocab)
│   └── tokenizer_config.json
├── src/
│   ├── model/                        # Custom Transformer, GQA, SwiGLU, RMSNorm, RoPE, EMA
│   ├── data/                         # Cleaner, deduplicator, document packing, shard reader
│   ├── training/                     # Trainer, optimizer, WSD scheduler, precision, checkpoints
│   ├── evaluation/                   # Loss, perplexity, KV-cache generator, benchmark probes
│   └── utils/                        # Device detection, config parser, structured logger
├── scripts/
│   ├── build_100m_corpus.py          # 100M token multi-domain corpus builder
│   ├── build_sft_dataset.py          # 10M token SFT dataset generator
│   ├── train.py                      # Main pretraining & continuation entrypoint
│   ├── train_sft.py                  # SFT training loop
│   ├── generate.py                   # Text generation CLI (KV-cache accelerated)
│   └── evaluate.py                   # Checkpoint evaluation harness
└── tests/                            # Comprehensive unit test suite (attention, model, rope, etc.)
```

---

## 6. Quickstart Guide

### 1. Environment Setup
```bash
git clone https://github.com/Asjad-Ilahi/huawei-slm.git
cd huawei-slm
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Architecture & System Capabilities
```bash
PYTHONPATH=. python scripts/inspect_model.py
```

### 3. Run Unit Tests
```bash
python -m unittest discover tests/
```

### 4. Text Generation with Checkpoint
```bash
PYTHONPATH=. python scripts/generate.py \
  --checkpoint experiments/exp_008/checkpoints/best.pt \
  --prompt "The scientific method begins with" \
  --max-tokens 100 \
  --temperature 0.7
```

### 5. Continuation Pretraining
To continue pretraining from the 237M-token milestone (`exp_008`) on new data:
```bash
PYTHONPATH=. python scripts/train.py \
  --config configs/model.yaml \
  --train-config configs/train_continuation.yaml \
  --data-dir data_100m/shards/train \
  --val-data-dir data_100m/shards/val \
  --resume experiments/exp_008/checkpoints/best.pt
```

---

## 7. Roadmap (Operation Apex)

* **Pretraining Target**: Scale continuation training to **1B – 2B tokens** on the frozen 54.5M architecture.
* **Reasoning Data Injection**: Expand verified arithmetic and chain-of-thought (CoT) traces to 20%+ of pretraining tokens.
* **SFT v2**: Integrate refusal and grounding training (rejection of false premises and computation limits) alongside AST-verified code and verified mathematics.
