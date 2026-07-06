import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-1.7B"
DATASET_NAME = "FinGPT/fingpt-sentiment-train"
OUTPUT_PATH = Path("data/formatted_sentiment.jsonl")

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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Total rows: {len(dataset):,}")

    written = skipped = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            formatted = format_row(row, tokenizer)
            if formatted is None:
                skipped += 1
                continue
            f.write(json.dumps({"text": formatted}) + "\n")
            written += 1

            if (i + 1) % 10_000 == 0:
                print(f"  Processed {i + 1:,} rows...")

    print(f"\nDone.")
    print(f"  Written : {written:,}")
    print(f"  Skipped : {skipped:,}")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    prepare()
