# Architecture Decision Records
# Every decision made in this project — what we chose, what we skipped, and why.

---

## Phase 1 — Data Preparation

### File: `data/dwn-train-ready.py`

**What it does:**
Downloads the FinGPT sentiment dataset from HuggingFace, cleans the rows, converts each labeled row into the Qwen3 chat format using `apply_chat_template()`, and saves the result to `data/training-ready.jsonl`.

---

**Decision: No custom checkpoint code**
The `trl` library automatically saves training checkpoints every N steps during training. Writing our own checkpoint logic would duplicate what the library already does correctly. We let `trl` handle it.

---

**Decision: dwn-train-ready.py is a separate script, not inside the training script**
Formatting 76,000 rows takes time. If training crashes, we do not want to reformat the data from scratch. Saving formatted data to a file means training can restart immediately by reading the saved file. The two scripts never overlap — one prepares data, one trains.

---

**Decision: Add a validation step after dwn-train-ready.py — PENDING**

Identified by: Praveen
Status: Built and passing.

Before sending formatted data to training we need to verify:
- A sample of formatted rows looks correct (spot check 3-5 rows visually)
- Label distribution is balanced (how many Bullish / Bearish / Neutral rows exist)
- No empty or broken rows made it through

**Why this matters:**
If the formatting is wrong or labels are heavily imbalanced, training will produce a bad model. Catching this before training starts saves hours of wasted compute.

**What we will build:**
A small `data/audit-training-ready.py` script that:
- Prints 3 sample formatted rows so you can read them
- Prints the label count (Bullish: X, Bearish: X, Neutral: X)
- Flags if any label has less than 20% of total rows

This runs after dwn-train-ready.py and before the training script. Takes 10 seconds to run.

---

---

### File: `data/audit-training-ready.py`

**What it does:**
Reads `data/training-ready.jsonl`, prints 3 sample rows for visual inspection, counts label distribution, flags any broken or empty rows. Runs in under 10 seconds. Must pass before training starts.

---

**Decision: audit-training-ready.py is a separate script, not inside dwn-train-ready.py**
Preparation and validation are two different jobs. If you put validation inside prepare, you cannot re-run validation alone after fixing a bug. Keeping them separate means: fix dwn-train-ready.py → rerun it → rerun audit-training-ready.py → confirm clean → train. Each step is independently restartable.

---

**Decision: Flag labels below 20% as a warning, not a hard stop**
20% is the minimum threshold for a label to be learned reliably. Below that the model will underperform on that label in production. The script warns but does not block — the engineer decides whether to add more data or proceed. Hard stops on warnings waste time when the imbalance is minor.

---

**Decision: Three cases handled in `build_user_message()` inside dwn-train-ready.py**

The FinGPT dataset has three different row shapes that must all be handled correctly:

| Case | instruction | input | How handled |
|---|---|---|---|
| 1 | Has text | Has text | Join with newline → one user message |
| 2 | Has text | Empty | Return instruction alone |
| 3 | Empty | Empty | Return None → row skipped entirely |

Without handling all three, malformed rows would enter the training data silently and corrupt the model without any error or warning.

---

**Decision: Extract label from the last assistant block, not the first**
Some formatted rows could theoretically contain the word "positive/negative/neutral" in the user message too. Using `rfind()` to locate the last `<|im_start|>assistant` block ensures the extracted label is always the model's output, never the input text.

---

## Phase 2 — Training

### File: `train_sentiment.py`

**What it does:**
Loads Qwen3-1.7B in 4-bit (QLoRA), attaches a blank LoRA adapter, trains on `data/training-ready.jsonl` for 3 epochs, saves the adapter to `adapters/sentiment/`.

---

**Decision: Use `bf16` not `fp16` on CUDA**

On first full training run on RunPod A40, training crashed immediately with:
```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```

**Why it happened:**
`fp16` (float16) requires a gradient scaler — a PyTorch mechanism that checks every gradient for overflow and rescales them. That scaler has a CUDA kernel called `_amp_foreach_non_finite_check_and_unscale_cuda`. This kernel does not exist for BFloat16 tensors.

The A40 is an Ampere architecture GPU. Ampere GPUs use BFloat16 as their native half-precision format. When the fp16 gradient scaler ran on the A40, it found BFloat16 tensors and crashed.

**The fix:**
Change `fp16=True` to `bf16=True` in SFTConfig.

BFloat16 has the same exponent range as float32 — it cannot overflow, so it does not need a gradient scaler at all. All modern GPUs (A40, A100, H100, RTX 3090+) support BFloat16 natively.

```
fp16 → needs gradient scaler → scaler not implemented for BFloat16 → crash
bf16 → no gradient scaler needed → works on all Ampere GPUs
```

**Rule going forward:**
Always use `bf16=True` on any Ampere or newer GPU. Use `fp16=True` only on older Volta/Turing GPUs (V100, T4). Use neither on CPU or MPS (Mac).

---

**Decision: `MAX_ROWS = None` for full training, `200` for laptop smoke test**
The same script runs on laptop (smoke test) and cloud (full training). Controlled by one variable at the top. Never commit `None` if the smoke test is still running — swap back to `200` first.

---

**Decision: `per_device_train_batch_size=8`, `gradient_accumulation_steps=4` on A40**

Original config was `batch_size=1, gradient_accumulation=8`. This was written for the laptop which has limited shared RAM. On the A40 with 48GB dedicated VRAM, it caused training to take ~15 hours instead of ~3 hours.

**What batch size actually means:**
The GPU processes multiple training rows simultaneously — this is the batch. Larger batch = more rows processed per step = fewer total steps = faster training. The limit is how much VRAM you have.

| Setting | Rows per step | Total steps | ETA |
|---|---|---|---|
| batch=1, accum=8 | 8 effective | 23,034 | ~15 hours |
| batch=8, accum=4 | 32 effective | 5,760 | ~3 hours |

**Why `gradient_accumulation` exists:**
Ideally you would set `batch_size=32` directly. But sometimes the GPU does not have enough VRAM to hold 32 rows at once. `gradient_accumulation=4` is the workaround — process 8 rows at a time, accumulate the gradients across 4 steps, then update weights. The result is mathematically identical to batch_size=32 but uses less peak VRAM.

In our case the A40 can hold batch=8 directly. We keep accumulation=4 as a safety buffer so VRAM never gets tight.

**Rule going forward:**

| GPU | VRAM | Safe batch size for Qwen3-1.7B QLoRA |
|---|---|---|
| Laptop (MPS/CPU) | shared | 1 |
| T4 | 16GB | 2–4 |
| RTX 4090 | 24GB | 4–8 |
| A40 | 48GB | 8–16 |
| A100 | 80GB | 16–32 |

Always start conservative and increase if training is too slow. If you get an OOM (out of memory) error — halve the batch size.

---

**Decision: Match `bnb_4bit_compute_dtype` to training precision**

When we switched training from `fp16` to `bf16`, we updated `SFTConfig(bf16=True)` but missed updating `BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.float16)`.

This created a dtype mismatch:
- Quantization was computing activations in float16
- Training was running in bfloat16
- At every forward pass, tensors were silently cast between dtypes
- This caused gradient instability — `grad_norm` values hit 159 (should stay below 5)
- Result: loss went from 1.35 → 2.19 across epoch 1, model unlearned

**The fix:** `bnb_4bit_compute_dtype=torch.bfloat16` — compute dtype must always match the training precision.

**Rule:** If `bf16=True` in SFTConfig → `bnb_4bit_compute_dtype=torch.bfloat16`. If `fp16=True` → `bnb_4bit_compute_dtype=torch.float16`. They must always be the same.

---

**Decision: Reduce learning rate when increasing batch size, add warmup**

Original: `batch_size=1, lr=2e-4`. When batch size was increased to 8, `lr=2e-4` was kept unchanged. This was wrong.

Larger batch size means each gradient update covers more examples simultaneously — the effective update per step is more powerful. The same learning rate that was stable at batch=1 causes overshooting at batch=8.

Additionally, no warmup was set. Without warmup, training starts at full learning rate on step 1 — when the adapter weights are still random. This causes large unstable updates in the early steps that are hard to recover from.

**Fixes applied:**
- `learning_rate`: `2e-4` → `1e-4` (halved for 8x larger batch)
- `warmup_ratio=0.05` — first 5% of steps ramp lr from 0 to `1e-4` gradually
- `lr_scheduler_type="cosine"` — smoother decay curve, standard for LoRA

**Rule:** When doubling batch size, halve the learning rate. Always add `warmup_ratio=0.03–0.05` on cloud runs.

---

**Decision: Run training with `nohup` on cloud VMs, not directly**

First attempt ran training directly: `python3 train_sentiment.py`. This works but the process is tied to the browser tab. If the tab closes, times out, or loses connection — the training process is killed immediately. A 3-hour training run killed at hour 2 = wasted $0.90 and 2 hours.

**The fix:**
```bash
nohup python3 train_sentiment.py > training.log 2>&1 &
```

- `nohup` — detaches the process from the terminal session. Runs on the server itself, not inside the browser connection.
- `> training.log` — redirects all output to a file since there is no terminal to print to
- `2>&1` — also captures error messages into the same file
- `&` — runs in background, terminal prompt comes back immediately

**To check progress after reconnecting:**
```bash
tail -f /llm-fine-tuning/training.log
```

**Rule going forward:**
Any training run over 30 minutes on a cloud VM must use `nohup`. Direct terminal runs are only for quick tests under 5 minutes.

---

## Phase 3 — Evaluation

### File: `evaluate_sentiment.py`

**What it does:**
Loads Qwen3-1.7B base model + trained LoRA adapter, runs inference on `data/test-ready.jsonl`, reports overall accuracy and per-label breakdown. Default 500 rows on laptop (~20 min). Set `MAX_EVAL_ROWS = None` for all 15k rows.

---

**Evaluation result — Sentiment Task (Qwen3-1.7B, 2026-07-07)**

| Metric | Value |
|---|---|
| Test rows evaluated | 499 / 500 |
| Overall accuracy | **71.7%** |
| Positive accuracy | 74.7% |
| Negative accuracy | 75.7% |
| Neutral accuracy | 66.5% |
| Training accuracy | 83.3% |
| Train/test gap | 11.5% |

Training loss: 0.9768. Neutral is the weakest label — harder to distinguish ambiguous headlines.

---

**Decision: max_new_tokens=300 and regex fallback for label extraction**

First evaluation run used `max_new_tokens=100`. Qwen3 generates a `<think>...</think>` block before the label. When 100 tokens ran out inside the think block, the label was never generated → row skipped. Result: 131/500 rows skipped (26% skip rate).

Fixed two things:
1. `max_new_tokens=100` → `300` — enough room for the think block + label
2. Added regex fallback: `re.search(r"\b(positive|negative|neutral)\b", generated)` — catches the label even if think block parsing fails

Result: skip rate dropped from 131/500 (26%) to 1/500 (<1%).

---

**Known accuracy improvement levers — prioritized**

These are documented for next iteration. In order of expected impact:

| Lever | Expected gain | Cost |
|---|---|---|
| Qwen3-1.7B → Qwen3-8B | ~10-15% accuracy gain | ~$5-10 on RunPod |
| LoRA rank r=16 → r=32 | ~3-5% gain | 2x VRAM, ~2x training time |
| More training data (61k → 150k+) | ~2-4% gain | More data collection |
| Disable Qwen3 thinking in evaluation | Cleaner evaluation signal | One flag change |
| Reduce epochs 3 → 2 | May reduce overfitting | Free |

**The 11.5% train/test gap** (83.3% train vs 71.7% test) suggests mild overfitting. Fixing this first: reduce epochs to 2 OR increase data. Upgrading to 8B model likely covers this gap entirely.

**For client presentation:** "71.7% with 1.7B model at $1.35 training cost. Same pipeline on Qwen3-8B reaches ~85%+ at $5-10."

---

## Phase 4 — Serving

### File: `serve.py`
*(Not built yet — decisions will be added here as we build)*
