# Finance LLM Fine-Tuning Project
## Financial Intelligence Assistant — Production Grade

---

## What This Project Is

Fine-tune a small open-source LLM (Qwen 3 8B) on public financial datasets to
build a Financial Intelligence Assistant that does 3 tasks banks actually use:

1. **Sentiment Analysis** — Read financial news → Bullish / Bearish / Neutral
2. **Earnings Call Summarization** — 60-page transcript → 5 bullet summary
3. **SEC Filing Q&A** — Ask any question → get the exact answer from the filing

This is what JPMorgan, Goldman Sachs, and Morgan Stanley are building internally.
You are building the open-source version of it, learning every concept from scratch.

---

## Why Finance

- Highest paying industry for AI engineers right now
- JPMorgan deployed this for 230,000 employees (LLM Suite)
- Goldman Sachs fine-tuned Llama on internal financial data
- A fine-tuned small model beats GPT-4 on narrow finance tasks at 1/500th the cost
- Public datasets exist — no proprietary data needed

---

## The Model

| Environment | Model | Why |
|---|---|---|
| Your laptop (learning) | Qwen 3 1.7B | Fits in 4GB RAM, same process |
| Client infrastructure | Qwen 3 8B or 30B | Swap one line, everything else identical |

**License:** Apache 2.0 — clean commercial use, no restrictions.

**Why Qwen 3 over Llama:**
- Newer and stronger than Llama 3.1 at same size
- Apache 2.0 is cleaner than Llama's custom license
- Better structured reasoning for financial tasks
- Full HuggingFace + vLLM support

**Why not GLM-5.2 or Gemma 4 31B:**
- GLM-5.2 = 744B params — cannot fine-tune on any single GPU
- Gemma 4 31B = tight on RTX 4090 (24GB), hard on laptop
- Both are deployment models, not fine-tuning models at consumer scale
- You can deploy them later once you understand the process

---

## Datasets (All Free, All Public)

```
1. AI4Finance-Foundation/fingpt-sentiment-train
   - 76,000 financial news headlines with sentiment labels
   - HuggingFace: datasets.load_dataset("AI4Finance-Foundation/fingpt-sentiment-train")

2. financial_phrasebank
   - 4,845 sentences labeled by finance professionals
   - HuggingFace: datasets.load_dataset("financial_phrasebank", "sentences_allagree")

3. nickmuchi/financial-classification
   - Earnings call sentences with classifications
   - HuggingFace: datasets.load_dataset("nickmuchi/financial-classification")

4. ProsusAI/fiqa
   - Financial QA pairs (questions + answers from financial forums)
   - HuggingFace: datasets.load_dataset("ProsusAI/fiqa")

5. SEC EDGAR (edgar.sec.gov)
   - 10-K and 10-Q filings for any public company
   - Fully public, no API key needed
   - Use for generating QA pairs on real filings
```

---

## Key Concepts You Will Learn

### Fine-Tuning
- **What is QLoRA?** Quantize the base model to 4-bit, then train only small adapter
  weights. Reduces GPU memory from 160GB (full fine-tune) to 5GB (QLoRA).
- **What is LoRA?** Instead of updating all 7B weights, train two small matrices A and B.
  Weight update = A×B. 99% fewer trainable parameters.
- **What is catastrophic forgetting?** Fine-tuning can overwrite general knowledge.
  LoRA prevents this — base weights stay frozen.
- **Multi-task fine-tuning** — training one model on 3 different tasks using
  instruction format so it learns all 3 without forgetting any.
- **When to fine-tune vs RAG?** RAG = knowledge retrieval. Fine-tune = style, format,
  domain terminology, consistent output structure, latency/cost matters.

### MLOps
- **Experiment tracking** — MLflow logs every training run: hyperparams, loss, metrics.
  5 runs with different settings, pick the best.
- **Model drift** — production inputs shift from training distribution, accuracy silently
  degrades. You detect it and auto-retrain.
- **CI/CD for ML** — GitHub Actions runs tests on every PR, deploys on merge, triggers
  retraining when drift is detected.

### Inference
- **vLLM** — 5x faster inference than raw HuggingFace. Uses PagedAttention (KV cache
  managed like OS virtual memory).
- **A/B testing** — route 50% traffic to fine-tuned model, 50% to GPT-4o, measure
  which wins on accuracy and cost before full rollout.

---

## Full Project Structure

```
finance-llm-finetune/
│
├── README.md                          ← This file
├── .env.example                       ← All environment variables listed here
├── .gitignore
├── requirements.txt
├── docker-compose.yml                 ← vLLM + FastAPI + Prometheus + Grafana
│
├── data/
│   ├── prepare_sentiment.py           ← Download FinGPT sentiment dataset, format it
│   ├── prepare_earningscalls.py       ← Download earnings call dataset, format it
│   ├── prepare_secqa.py               ← Pull SEC filings, generate QA pairs
│   └── merge_datasets.py              ← Combine all 3 into one instruction dataset
│
├── training/
│   ├── finetune.py                    ← QLoRA training (main script)
│   ├── config.yaml                    ← Hyperparams: lr, batch_size, epochs, lora_r
│   ├── evaluate.py                    ← BLEU, BERTScore, accuracy per task on test set
│   ├── push_to_hub.py                 ← Upload best model to HuggingFace Hub
│   └── mlflow_logger.py              ← Log all metrics to MLflow during training
│
├── serving/
│   ├── vllm_server.py                 ← Spin up vLLM inference server
│   ├── openai_client.py              ← GPT-4o client (control arm for A/B test)
│   └── model_loader.py               ← Load fine-tuned model from HuggingFace Hub
│
├── api/
│   ├── main.py                        ← FastAPI app
│   └── routes/
│       ├── sentiment.py               ← POST /sentiment
│       ├── summarize.py               ← POST /summarize
│       ├── qa.py                      ← POST /ask
│       ├── ab_test.py                 ← A/B traffic routing logic
│       └── health.py                  ← GET /health
│
├── monitoring/
│   ├── prometheus_metrics.py          ← Define metrics: latency, accuracy, cost
│   ├── grafana_dashboard.json         ← Import into Grafana
│   ├── drift_detector.py              ← Evidently AI drift detection per task
│   └── alert_rules.yml               ← Alert if sentiment accuracy < 0.82
│
├── retraining/
│   └── retrain_trigger.py            ← Called by GitHub Actions when drift detected
│
├── kubernetes/
│   ├── vllm-deployment.yaml
│   ├── api-deployment.yaml
│   └── prometheus-configmap.yaml
│
├── tests/
│   ├── test_data.py                   ← Verify dataset loading and tokenization
│   ├── test_sentiment.py              ← Verify sentiment classification
│   ├── test_summarization.py         ← Verify summarization output
│   ├── test_qa.py                     ← Verify QA responses
│   └── test_api.py                    ← Verify all API endpoints
│
└── .github/
    └── workflows/
        ├── ci.yml                     ← Run tests on every PR
        ├── deploy.yml                 ← Deploy to K8s on push to main
        └── retrain.yml               ← Triggered by drift alert → run finetune.py
```

---

## Fine-Tuning Details

### Why QLoRA and not full fine-tuning?
Full fine-tuning of Qwen 3 8B requires ~160GB VRAM.
QLoRA quantizes the base model to 4-bit (~5GB) and trains only small LoRA adapter
weights (~80MB). Same quality, 96% less GPU memory. You can run this on a laptop.

### Instruction Format (What Your Training Data Looks Like)

```
# Sentiment task
<|user|>
Analyze the sentiment of this financial news:
"Apple reports record quarterly earnings, beats analyst expectations by 15%"
<|assistant|>
Bullish

# Summarization task
<|user|>
Summarize this earnings call transcript in 5 bullet points:
[transcript text]
<|assistant|>
- Revenue grew 12% YoY to $89.5B, driven by iPhone and Services
- Gross margin expanded to 46.2%, highest in company history
- Management guided Q2 revenue of $85-88B, below consensus $91B
- Services segment now $23.8B quarterly run rate, growing 17% YoY
- Share buyback program increased by $110B

# QA task
<|user|>
Based on this SEC 10-K filing, what are the top 3 risk factors?
[filing text]
<|assistant|>
1. Macroeconomic conditions...
2. Regulatory compliance...
3. Cybersecurity threats...
```

### Hyperparameter Experiments (5 MLflow Runs)

```
Run 1: lr=2e-4, batch=4, epochs=3, lora_r=16   ← baseline
Run 2: lr=1e-4, batch=4, epochs=3, lora_r=16   ← lower lr
Run 3: lr=2e-4, batch=8, epochs=3, lora_r=16   ← larger batch
Run 4: lr=2e-4, batch=4, epochs=5, lora_r=16   ← more epochs
Run 5: lr=2e-4, batch=4, epochs=3, lora_r=32   ← larger lora rank
```

MLflow tracks: loss, accuracy per task, BERTScore, training time, GPU memory.
Best model gets pushed to HuggingFace Hub.

---

## Hardware

| Stage | Machine | Cost |
|---|---|---|
| Learning | Your laptop (Qwen 3 1.7B) | Free |
| Full training | RunPod RTX 4090 (Qwen 3 8B) | ~$0.44/hr |
| Client infra | Whatever the client has | They pay |

When client has infrastructure → change one line:
```python
# Laptop
model_name = "Qwen/Qwen3-1.7B"

# Client GPU cluster
model_name = "Qwen/Qwen3-8B"   # or 30B, 72B — same code, same pipeline
```

---

## A/B Test — What You Are Measuring

| Metric | Fine-tuned Qwen 3 8B | GPT-4o |
|---|---|---|
| Avg latency | ~45ms (vLLM) | ~800ms (API) |
| Cost per 1000 requests | ~$0.02 | ~$10 |
| Sentiment accuracy | Measure | Measure |
| Summarization BERTScore | Measure | Measure |

Goal: fine-tuned model matches GPT-4o on accuracy at 1/500th the cost.
This is the exact business case you present to a finance client.

---

## Environment Variables

```bash
# HuggingFace
HF_TOKEN=
HF_MODEL_REPO=your-username/qwen3-finance-finetuned

# OpenAI (for Model B in A/B test)
OPENAI_API_KEY=

# MLflow
MLFLOW_TRACKING_URI=
MLFLOW_EXPERIMENT_NAME=qwen3-finance-finetune

# RunPod (for full training)
RUNPOD_API_KEY=

# Database (A/B result logging)
DATABASE_URL=postgresql://user:pass@localhost/finance_llm

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

---

## Phase-by-Phase Build Plan

### Phase 1 — Data + First Training Run (Week 1)
- [ ] Install dependencies: transformers, peft, trl, bitsandbytes, datasets
- [ ] Run `prepare_sentiment.py` — download FinGPT dataset, verify it loads
- [ ] Format into instruction template, tokenize with Qwen 3 tokenizer
- [ ] Run first training job on Qwen 3 1.7B (laptop, just to see it work)
- [ ] Verify loss goes down, model produces sensible outputs
- [ ] **Milestone:** Model trained on sentiment task, running on laptop

### Phase 2 — Multi-Task + MLflow (Week 2)
- [ ] Add earnings call and SEC QA datasets
- [ ] Merge all 3 into one instruction dataset
- [ ] Set up MLflow tracking (local or Dagshub)
- [ ] Run all 5 hyperparameter experiments, compare in MLflow UI
- [ ] Evaluate best model: accuracy on each of the 3 tasks
- [ ] Push best model to HuggingFace Hub
- [ ] **Milestone:** Multi-task model on HuggingFace Hub, 5 MLflow runs compared

### Phase 3 — Serving + A/B Test (Week 3)
- [ ] Install vLLM, load fine-tuned model, benchmark tokens/second
- [ ] Build FastAPI with /sentiment, /summarize, /ask endpoints
- [ ] Wire GPT-4o as Model B (control arm)
- [ ] Build A/B routing: hash(user_id) → Model A or B
- [ ] Dockerize API + vLLM
- [ ] **Milestone:** Both models serving behind one API, A/B routing working

### Phase 4 — Monitoring + Drift + CI/CD (Week 4)
- [ ] Set up Prometheus + Grafana via Docker Compose
- [ ] Build drift detector: compare live sentiment distribution vs baseline
- [ ] Set alert rule: sentiment accuracy < 0.82 → trigger retraining
- [ ] Write GitHub Actions CI (test on PR) + deploy (push to main)
- [ ] Write retrain.yml — drift alert → runs finetune.py automatically
- [ ] Test full loop: degrade inputs → drift detected → retrain → new model live
- [ ] **Milestone:** Full automated ML lifecycle end-to-end

---

## Interview Questions This Project Answers

**ML System Design:**
- "Design a sentiment analysis system for financial news at scale" → you built it
- "How do you fine-tune an LLM for a specific domain cheaply?" → QLoRA, <$300
- "How do you handle multiple NLP tasks with one model?" → multi-task instruction tuning
- "Design an A/B testing system for ML models" → you built one
- "How do you detect and respond to model drift in production?" → Evidently AI + auto-retrain

**Finance Domain:**
- "What is the difference between sentiment analysis and summarization as ML tasks?" → classification vs generation, different evaluation metrics
- "How would you build the JPMorgan LLM Suite?" → this project is the open-source version of it
- "Why fine-tune instead of just using GPT-4?" → cost, latency, data privacy, domain accuracy

**MLOps:**
- "How do you track ML experiments?" → MLflow, 5 runs, compared loss + accuracy + BERTScore
- "What is your CI/CD pipeline for ML?" → test → build → deploy → monitor → retrain

---

## Success Metrics

| Metric | Target |
|---|---|
| Sentiment accuracy vs GPT-4o | Within 5% |
| Summarization BERTScore | > 0.82 |
| vLLM latency vs raw HuggingFace | 5x improvement |
| Cost per 1000 requests | < $0.05 |
| Drift detection time | < 1 hour |
| Retrain pipeline: drift → model live | < 4 hours |

---

## Status

**Current phase:** Not started
**Next action when ready:** Install dependencies and run `prepare_sentiment.py`

```bash
pip install transformers peft trl bitsandbytes datasets mlflow fastapi uvicorn
```

Then open `data/prepare_sentiment.py` and run it. That is step one.

---

## References

- [FinGPT — Open-Source Financial LLMs](https://github.com/AI4Finance-Foundation/FinGPT)
- [FinLoRA — Benchmarking LoRA on Finance Datasets](https://arxiv.org/pdf/2505.19819)
- [JPMorgan LLM Suite](https://thedigitalbanker.com/jpmorgan-chases-llm-suite-drives-ai-transformation-across-the-enterprise/)
- [Qwen 3 on HuggingFace](https://huggingface.co/Qwen)
- [Evidently AI — Drift Detection](https://www.evidentlyai.com/)
- [vLLM — Fast Inference](https://github.com/vllm-project/vllm)
- [MLflow — Experiment Tracking](https://mlflow.org/)
