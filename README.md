# Finance LLM — Domain Fine-Tuning with QLoRA

Fine-tuned **Qwen3-1.7B** on public financial datasets using QLoRA. Three production tasks: sentiment classification, earnings call summarization, and SEC filing Q&A. Full pipeline from raw data to evaluated adapter.

---

## Results

| Task | Model | Accuracy / Score | Test Set |
|---|---|---|---|
| Sentiment Analysis | Qwen3-1.7B + QLoRA | **71.7%** | 499 held-out headlines |
| Earnings Summarization | In progress | BERTScore target > 0.82 | — |
| SEC Filing Q&A | Planned | — | — |

Trained on RunPod A40 (48GB VRAM). Full sentiment training: 2h 39m, $1.35.

---

## Architecture

```
Raw Dataset (HuggingFace)
        ↓
dwn-train-ready.py       — download, normalize labels, format to Qwen3 chat template
        ↓
audit-training-ready.py  — verify label distribution, catch broken rows before training
        ↓
train_sentiment.py       — QLoRA training: 4-bit base + LoRA adapter, 3 epochs
        ↓
evaluate_sentiment.py    — accuracy on held-out test set (20% split, never seen in training)
        ↓
adapters/sentiment/      — saved LoRA adapter (~80MB), loads on top of any Qwen3 base
```

---

## Technical Decisions

**QLoRA over full fine-tuning**
Full fine-tuning of Qwen3-8B requires ~160GB VRAM. QLoRA quantizes the base model to 4-bit (~5GB) and trains only a small LoRA adapter (~80MB). Same output quality, 96% less GPU memory.

**Three separate adapters, one base model**
Each task gets its own adapter. The base model is frozen and shared. Swapping tasks = swapping adapter file, not reloading a 7GB model.

**80/20 train/test split**
FinGPT has no official test split. We split before training with `random.seed(42)` — reproducible across machines. Test set is locked and never touched during training.

**Why Qwen3 over Llama**
Apache 2.0 license (cleaner than Llama's custom license). Stronger reasoning at equivalent size. Full HuggingFace and vLLM support.

---

## Training Config

```python
model                    = "Qwen/Qwen3-1.7B"
lora_r                   = 16
lora_alpha               = 32
target_modules           = ["q_proj", "v_proj"]
per_device_train_batch   = 8
gradient_accumulation    = 4     # effective batch = 32
learning_rate            = 1e-4
warmup_ratio             = 0.05
lr_scheduler             = "cosine"
epochs                   = 3
bf16                     = True
bnb_4bit_compute_dtype   = bfloat16   # must match bf16 training precision
```

---

## Datasets

| Task | Dataset | Size | Source |
|---|---|---|---|
| Sentiment | FinGPT/fingpt-sentiment-train | 76,772 rows | HuggingFace |
| Summarization | TBD | — | — |
| SEC Q&A | ProsusAI/fiqa | — | HuggingFace |

All public, no API keys required.

---

## Project Structure

```
llm-fine-tuning/
├── data/
│   ├── dwn-train-ready.py        ← download + normalize + format to Qwen3 chat template
│   ├── audit-training-ready.py   ← verify formatted data before training
│   ├── training-ready.jsonl      ← 61,417 training rows (gitignored)
│   └── test-ready.jsonl          ← 15,355 held-out test rows (gitignored)
├── adapters/
│   └── sentiment/                ← trained LoRA adapter (gitignored)
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       ├── checkpoint-1920/      ← epoch 1
│       ├── checkpoint-3840/      ← epoch 2
│       └── checkpoint-5760/      ← epoch 3 (final)
├── train_sentiment.py            ← QLoRA training script
├── evaluate_sentiment.py         ← accuracy evaluation on held-out test set
├── decisions.md                  ← architecture decisions + real-world bugs fixed
└── requirements.txt
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Prepare data**
```bash
USE_TF=0 python3 data/dwn-train-ready.py
USE_TF=0 python3 data/audit-training-ready.py
```

**3. Train (RunPod A40 recommended)**
```bash
nohup python3 train_sentiment.py > training.log 2>&1 &
tail -f training.log
```

**4. Evaluate**
```bash
USE_TF=0 python3 evaluate_sentiment.py
```

---

## Status

| Component | Status |
|---|---|
| Sentiment data pipeline | ✅ Complete |
| Sentiment training | ✅ Complete — 71.7% accuracy on held-out test |
| Earnings call pipeline | 🔄 In progress |
| SEC filing pipeline | ⏳ Planned |
| vLLM serving + FastAPI | ⏳ Planned |
| A/B test vs GPT-4o | ⏳ Planned |
| MLflow experiment tracking | ⏳ Planned |

---

## References

- [FinGPT — Open-Source Financial LLMs](https://github.com/AI4Finance-Foundation/FinGPT)
- [FinLoRA — Benchmarking LoRA on Finance Datasets](https://arxiv.org/pdf/2505.19819)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Qwen3 on HuggingFace](https://huggingface.co/Qwen)
- [vLLM — Fast LLM Inference](https://github.com/vllm-project/vllm)
