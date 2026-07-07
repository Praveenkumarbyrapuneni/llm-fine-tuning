import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME    = "Qwen/Qwen3-1.7B"
DATASET_NAME  = "FinGPT/fingpt-sentiment-train"
TRAIN_PATH    = Path("data/training-ready.jsonl")
TEST_PATH     = Path("data/test-ready.jsonl")
TEST_SPLIT    = 0.2   # 20% held out, never seen during training
RANDOM_SEED   = 42

SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. "
    "Classify the sentiment of the given financial text. "
    "Reply with exactly one word: positive, negative, or neutral."
)

# FinGPT dataset has 9 label variants — normalize all to 3
LABEL_MAP = {
    "positive":           "positive",
    "moderately positive": "positive",
    "mildly positive":    "positive",
    "strong positive":    "positive",
    "negative":           "negative",
    "moderately negative": "negative",
    "mildly negative":    "negative",
    "strong negative":    "negative",
    "neutral":            "neutral",
}


def build_user_message(row: dict) -> str:
    instruction = row.get("instruction", "").strip()
    input_text = row.get("input", "").strip()
    if input_text:
        return f"{instruction}\n{input_text}"
    return instruction


def format_row(row: dict, tokenizer: AutoTokenizer) -> str | None:
    user_message = build_user_message(row)
    raw_output = row.get("output", "").strip().lower()
    output = LABEL_MAP.get(raw_output)

    if not user_message or output is None:
        return None

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def prepare() -> None:
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Total rows: {len(dataset):,}")

    # Format all valid rows first
    formatted_rows = []
    skipped = 0
    for i, row in enumerate(dataset):
        formatted = format_row(row, tokenizer)
        if formatted is None:
            skipped += 1
            continue
        formatted_rows.append(json.dumps({"text": formatted}))
        if (i + 1) % 10_000 == 0:
            print(f"  Processed {i + 1:,} rows...")

    # Shuffle then split 80/20 — fixed seed so split is always identical
    random.seed(RANDOM_SEED)
    random.shuffle(formatted_rows)
    split_idx = int(len(formatted_rows) * (1 - TEST_SPLIT))
    train_rows = formatted_rows[:split_idx]
    test_rows  = formatted_rows[split_idx:]

    TRAIN_PATH.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    TEST_PATH.write_text("\n".join(test_rows) + "\n", encoding="utf-8")

    print(f"\nDone.")
    print(f"  Total formatted : {len(formatted_rows):,}")
    print(f"  Skipped         : {skipped:,}")
    print(f"  Training rows   : {len(train_rows):,}  → {TRAIN_PATH}")
    print(f"  Test rows       : {len(test_rows):,}   → {TEST_PATH}")


if __name__ == "__main__":
    prepare()
