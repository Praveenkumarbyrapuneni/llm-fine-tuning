# Architecture Decision Records
# Every decision made in this project — what we chose, what we skipped, and why.

---

## Phase 1 — Data Preparation

### File: `data/prepare_sentiment.py`

**What it does:**
Downloads the FinGPT sentiment dataset from HuggingFace, cleans the rows, converts each labeled row into the Qwen3 chat format using `apply_chat_template()`, and saves the result to `data/formatted_sentiment.jsonl`.

---

**Decision: No custom checkpoint code**
The `trl` library automatically saves training checkpoints every N steps during training. Writing our own checkpoint logic would duplicate what the library already does correctly. We let `trl` handle it.

---

**Decision: prepare_sentiment.py is a separate script, not inside the training script**
Formatting 76,000 rows takes time. If training crashes, we do not want to reformat the data from scratch. Saving formatted data to a file means training can restart immediately by reading the saved file. The two scripts never overlap — one prepares data, one trains.

---

**Decision: Add a validation step after prepare_sentiment.py — PENDING**

Identified by: Praveen
Status: Not built yet — will be built after prepare_sentiment.py is complete.

Before sending formatted data to training we need to verify:
- A sample of formatted rows looks correct (spot check 3-5 rows visually)
- Label distribution is balanced (how many Bullish / Bearish / Neutral rows exist)
- No empty or broken rows made it through

**Why this matters:**
If the formatting is wrong or labels are heavily imbalanced, training will produce a bad model. Catching this before training starts saves hours of wasted compute.

**What we will build:**
A small `data/validate_sentiment.py` script that:
- Prints 3 sample formatted rows so you can read them
- Prints the label count (Bullish: X, Bearish: X, Neutral: X)
- Flags if any label has less than 20% of total rows

This runs after prepare_sentiment.py and before the training script. Takes 10 seconds to run.

---

---

### File: `data/validate_sentiment.py`

**What it does:**
Reads `data/formatted_sentiment.jsonl`, prints 3 sample rows for visual inspection, counts label distribution, flags any broken or empty rows. Runs in under 10 seconds. Must pass before training starts.

---

**Decision: validate_sentiment.py is a separate script, not inside prepare_sentiment.py**
Preparation and validation are two different jobs. If you put validation inside prepare, you cannot re-run validation alone after fixing a bug. Keeping them separate means: fix prepare → rerun prepare → rerun validate → confirm clean → train. Each step is independently restartable.

---

**Decision: Flag labels below 20% as a warning, not a hard stop**
20% is the minimum threshold for a label to be learned reliably. Below that the model will underperform on that label in production. The script warns but does not block — the engineer decides whether to add more data or proceed. Hard stops on warnings waste time when the imbalance is minor.

---

**Decision: Three cases handled in `build_user_message()` inside prepare_sentiment.py**

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
*(Not built yet — decisions will be added here as we build)*

---

## Phase 3 — Evaluation

### File: `evaluate_sentiment.py`
*(Not built yet — decisions will be added here as we build)*

---

## Phase 4 — Serving

### File: `serve.py`
*(Not built yet — decisions will be added here as we build)*
