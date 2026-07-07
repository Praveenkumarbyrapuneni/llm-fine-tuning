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

## Phase 3 — Evaluation

### File: `evaluate_sentiment.py`
*(Not built yet — decisions will be added here as we build)*

---

## Phase 4 — Serving

### File: `serve.py`
*(Not built yet — decisions will be added here as we build)*
