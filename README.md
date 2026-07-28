# Arabic Content Moderation

A two-stage classifier cascade for **Egyptian-dialect Arabic** — dialect-specific
categories from the first stage, backed by a multilingual toxicity model that catches the
implicit hostility the first stage misses. Returns a category, a recommended action, an
Arabic explanation, and optional word-level redaction, over a FastAPI service.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="Transformers" src="https://img.shields.io/badge/%F0%9F%A4%97-Transformers-FFD21E">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-36%20passing-16A34A">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Why two stages

The original research found that the dialect hate-speech model *"detects hate/offensive
speech well, but if the sentence does not contain a hate **word**, it's not detected."*

That is a **recall** gap on implicit hostility, not a precision problem — which points at a
specific fix. Stacking a broad multilingual toxicity model *behind* the dialect model
targets exactly that gap: stage 1 keeps its fine-grained category labels, and stage 2 is
only ever consulted on text stage 1 has already called clean.

An ensemble vote would have been the obvious alternative and the wrong one — averaging
would dilute stage 1's category information, which is the thing that makes the output
actionable.

```mermaid
flowchart TB
    T[Arabic text] --> S1[Stage 1: MARBERTv2<br/>Egyptian-dialect hate speech]
    S1 -->|harmful category| V[Verdict + category]
    S1 -->|Neutral| S2[Stage 2: XLM-R<br/>multilingual toxicity]
    S2 -->|score >= 0.60| VE[Verdict: Extremism]
    S2 -->|below threshold| VN[Verdict: Neutral]
    V --> A{Action}
    VE --> A
    VN --> A
    A -->|Offensive / Racism / Extremism| M[Mask harmful tokens]
    A -->|Sexism / Religious| F[Flag with explanation]
    A -->|Neutral| P[Allow]
```

## Results

36 hand-labelled Egyptian-dialect examples. The **stage-1-only ablation** is the number
that matters — without it, the cascade's score means nothing.

### Binary — harmful vs neutral

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Stage 1 only (MARBERTv2) | 0.778 | **0.947** | 0.720 | 0.818 |
| **Full cascade** | **0.806** | 0.846 | **0.880** | **0.863** |

**The second stage does exactly what it was added to do.** Recall climbs from 0.720 to
0.880 — it catches 4 harmful messages the dialect model let through. The cost is
precision, down from 0.947 to 0.846. Net F1 improves from 0.818 to 0.863.

Whether that trade is correct **depends on the platform**, and this is a policy decision
rather than a modelling one. A platform where missed abuse is the greater harm should take
the cascade. One where over-blocking drives users away should keep stage 1 alone, or raise
`TOXICITY_THRESHOLD` above its 0.60 default to buy back precision.

### Multi-class — 6 categories

| Configuration | Accuracy | Macro F1 |
|---|---:|---:|
| Stage 1 only | 0.722 | — |
| Full cascade | 0.722 | 0.783 |

**Identical**, and that is expected: stage 2 can only ever emit `Extremism`, so it adds
recall on the binary question without adding category granularity.

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Sexism | 1.00 | 1.00 | 1.00 | 3 |
| Religious Discrimination | 1.00 | 1.00 | 1.00 | 2 |
| Offensive | 0.88 | 0.88 | 0.88 | 8 |
| Racism | 0.67 | 0.80 | 0.73 | 5 |
| Neutral | 0.70 | 0.64 | 0.67 | 11 |
| **Extremism** | **0.43** | **0.43** | **0.43** | 7 |

Perfect scores on Sexism and Religious Discrimination rest on 3 and 2 examples
respectively. **They are not evidence of anything** — they are what a tiny support column
looks like.

### Where it fails

All 10 errors, because the pattern in them is more informative than the aggregate:

| Expected | Predicted | Text | What went wrong |
|---|---|---|---|
| Extremism | Neutral | اللي مش معانا يبقى عدو ولازم يتصفّى | "whoever is not with us is an enemy and must be liquidated" — **missed entirely** |
| Extremism | Neutral | ياريت الناس دي كلها تختفي من الدنيا | Eliminationist wish, phrased without a slur |
| Extremism | Neutral | اللي رأيه غير رأيي يبقى عدوى | Dehumanising framing, no explicit threat |
| Extremism | Offensive | لازم نكسر ونولّع فيهم | Incitement downgraded to insult |
| Neutral | Extremism | يجب قتل الحشرات الضارة | "pests must be killed" — **pest control** |
| Neutral | Extremism | عايزة أقتل الحشرات الضارة | Same trap, first person |
| Neutral | Extremism | المباراة كانت وحشة والفريق لعب سيء | Criticising a football match |
| Neutral | Racism | الراجل الاسود طبيب | "the black man is a doctor" — **neutral statement mentioning race** |
| Offensive | Racism | انت عبد وملكش قيمة | Category confusion; correctly caught as harmful |
| Racism | Extremism | الناس من البلد دي كلهم كذابين | Category confusion; correctly caught as harmful |

Three distinct failure modes:

1. **Implicit extremism is the real weakness** (recall 0.43). Every miss is a threat
   carried by *structure* — "must be eliminated", "should disappear" — rather than by a
   flaggable word. This is precisely the gap the cascade was meant to close, and it only
   partly does.
2. **Both models are keyword-triggered, not semantic.** قتل ("kill") fires on pest control;
   اسود ("black") fires on a sentence stating someone is a doctor. **The race false
   positive is the most serious error here** — a moderation system that flags neutral
   mentions of race will disproportionately silence the people it exists to protect.
3. **Category confusion is benign.** The last two rows are "wrong" but both were correctly
   identified as harmful and would be actioned. Multi-class accuracy understates
   deployment usefulness.

### Throughput

**0.39 s/example** end-to-end on CPU (36 examples in 14.2 s), including word-level masking.
17 of 36 inputs escalated to stage 2. Actions taken: 10 allow, 5 flag, 21 mask.

## What changed from the research notebook

**Low-confidence neutrals are handled.** The notebook branched on a bare
`label == "neutral"` string check, so a 0.34-confidence neutral was treated as settled.
Confidence now participates in routing.

**Stage 2 must clear a threshold to overturn stage 1.** The notebook accepted any
`LABEL_1`, including a 0.51 coin-flip. A broad multilingual model misreads dialect; making
it earn the overturn is what keeps that from becoming a false positive.

**Label normalisation is centralised.** The two models emit incompatible vocabularies —
readable category names with inconsistent casing versus `LABEL_0`/`LABEL_1`. That mapping
now lives in [`labels.py`](src/moderation/labels.py) with tests, instead of being
string-matched at each call site. Unknown labels **fail closed to `Offensive`**, never to
`Neutral`: if a model flagged something under a name we don't recognise, treating it as
clean is the more dangerous error.

**Masking has its own, higher threshold** (0.80 vs 0.60 to flag). Over-redacting a clean
word is visible and irritating; the sentence is already flagged, so the bar to redact
should exceed the bar to act.

**Sexism and religious discrimination are flagged, not masked.** Both are usually carried
by sentence structure rather than a removable word — masking produces a mutilated sentence
that still says the same thing.

## Quickstart

```bash
git clone https://github.com/yasmine-ali101/ai-content-moderation.git
cd ai-content-moderation

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/evaluate.py    # reproduces every number above
pytest                        # 36 tests, no model download needed
```

> First run downloads ~3 GB of weights. `sentencepiece` and `tiktoken` are required — XLM-R's
> tokenizer fails to load without them, with a fairly opaque error.

### As a library

```python
from moderation import ModerationPipeline

verdict = ModerationPipeline().moderate("انتي غبية")

verdict.category      # 'Offensive'
verdict.action        # 'mask'
verdict.masked_text   # 'انتي ****'
verdict.explanation   # Arabic rationale shown to the user
verdict.escalated     # False — stage 1 was confident
```

### As a service

```bash
uvicorn moderation.api:app --reload
```

```bash
curl -X POST localhost:8000/moderate \
     -H 'Content-Type: application/json' \
     -d '{"text": "انت غبي"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /moderate` | Single text |
| `POST /moderate/batch` | Up to 64 texts |
| `GET /health` | Readiness + active model names |

Models load once at startup, not per request — a cold `from_pretrained` costs tens of
seconds and would otherwise dominate every response.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `TOXICITY_THRESHOLD` | `0.60` | Raise to buy precision back from the cascade |
| `NEUTRAL_CONFIDENCE_FLOOR` | `0.85` | Below this, stage 1's neutral is treated as shaky |
| `MASK_THRESHOLD` | `0.80` | Word-level redaction confidence |
| `MODERATION_DEVICE` | `-1` | `-1` CPU, `0` first GPU |
| `HATE_MODEL` / `TOXICITY_MODEL` | see [`config.py`](src/moderation/config.py) | Swap either stage |

## Project structure

```
src/moderation/
├── config.py       # thresholds and model selection, env-driven
├── labels.py       # canonical vocabulary + fail-closed normalisation
├── pipeline.py     # the cascade, masking, and action policy
├── evaluation.py   # binary + multi-class scoring
└── api.py          # FastAPI service
scripts/evaluate.py # reproduces the results above
data/eval_set.json  # 36 hand-labelled examples
results/            # metrics.json, metrics.md (regenerated)
tests/              # 36 tests, all stubbed — no model download
```

## Limitations

These bound what the numbers above are worth:

- **The evaluation set has 36 examples and was annotated by the project authors.** No
  inter-annotator agreement, no held-out split. Per-category figures resting on 2-3
  examples (Sexism, Religious Discrimination) are illustrative, not measurements. Treat
  every number here as a smoke test, not a benchmark — a real deployment needs a
  few thousand examples labelled by multiple annotators.
- **Several examples are genuinely multi-label** ("انت كلب ومتخلف الستات مكانهم المطبخ" is
  both Offensive and Sexism) but are annotated with a single most-severe category. The
  multi-class score penalises the model for picking the other valid label.
- **Extremism recall is 0.43.** The system should not be relied on for threat detection.
- **Neutral precision is 0.70** — roughly a third of clean messages get flagged. That is a
  poor user experience and the first thing to fix.
- **No Modern Standard Arabic evaluation.** Both the eval set and stage 1 are
  Egyptian-dialect; behaviour on MSA or Gulf/Levantine dialects is untested.
- **Word-level masking is a blunt instrument.** These models were trained on sentences; a
  single word out of context is a distribution they never saw.
- **No adversarial testing** — character substitution, elongation (كككلب), or Arabizi
  transliteration would likely defeat it.

## Attribution

Built as a group project for the **BARQ** AI program by **Habiba, Yasmine Ali, and Aliaa**.
This repository is the productionised refactor — routing logic, thresholds, label
normalisation, evaluation harness, API, and tests — of our shared research notebook, which
is preserved in [`notebooks/`](notebooks/).

Models are third-party: [MARBERTv2 fine-tune](https://huggingface.co/IbrahimAmin/marbertv2-finetuned-egyptian-hate-speech-classification)
by Ibrahim Amin, and [xlm-r-large-arabic-toxic](https://huggingface.co/akhooli/xlm-r-large-arabic-toxic)
by Abed Khooli.

## License

[MIT](LICENSE)
