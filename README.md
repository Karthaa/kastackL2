# KaStack AI Intern — L2 Assessment

> Adaptive Persona Engine · Offline Intent Classifier · RAG Conflict Resolver · Sync Architecture

---

## Demo
**(https://kastack-l2.vercel.app/)**

🔗 (https://www.loom.com/share/3033c8f1ee21427693a6fc5a2d46bb02)

---

## What This Does

A four-part AI system built for the KaStack Labs L2 intern assessment. Each part is fully functional — no mocks, no hardcoded outputs.

| Part | What it does |
|------|-------------|
| 01 · Persona Drift Detector | Tracks tone/mood changes across conversation days, detects drift events, maps triggers |
| 02 · Offline Intent Classifier | TF-IDF + Naive Bayes trained in-browser, 3-fold CV, real probability output |
| 03 · RAG Conflict Resolver | Ranks chunks by recency + emotional weight, flags contradictions, merges answer |
| 04 · System Design | On-device storage, sync architecture, CRDT conflict resolution |

---

## Part 1 — Persona Drift Detector

Reads Alex's conversation history across 7 days and computes:

- **Tone profile per day** — formal, casual, curious, frustrated, playful, anxious, relieved
- **Mood trajectory** — positive / neutral / negative with sparkline chart
- **Formality score** — keyword ratio (formal vs casual language)
- **Drift detection** — compares consecutive day profiles, flags when formality or mood shifts
- **Trigger extraction** — regex scan for people (sister, mentor, sarah), events (deadline, submitted, mistake), topics (pytorch, transformer, gpt)

**Timeline output:**
```
Day 1 → formal + curious       triggers: gradient, evaluation
Day 2 → formal + curious       triggers: sister (person), transformer (topic)
Day 4 → casual + frustrated    DRIFT ⚡  triggers: sister (person), deadline (event)
Day 6 → casual + positive      DRIFT ⚡  triggers: submitted (event), sarah (person)
Day 7 → casual + playful       triggers: sister (person), mistake (event), diffusion (topic)
```

---

## Part 2 — Offline Intent Classifier

A real ML model trained entirely in the browser. No OpenAI, no Gemini, no server.

**Architecture:**
```
Input text
    ↓
Tokenizer (unigram + bigram)
    ↓
TF-IDF weighting  (IDF computed over full corpus)
    ↓
Multinomial Naive Bayes  (Laplace smoothing α=0.1)
    ↓
Softmax probabilities
    ↓
Intent label  (threshold: 0.25)
```

**Training data:** 78 samples across 5 classes

| Class | Samples |
|-------|---------|
| reminder | 15 |
| emotional-support | 18 |
| action-item | 15 |
| small-talk | 15 |
| unknown | 15 |

**Performance:**
- 3-fold cross-validation accuracy: ~90–95%
- Model size: ~40–60 KB (well under 50 MB limit)
- Inference latency: < 10ms on CPU (well under 200ms limit)

---

## Part 3 — RAG Conflict Resolver

**The problem:** User asks *"Did I mention anything about my sister?"*
The word "sister" appears in 3 chunks with contradictory emotional context.

**Scoring formula:**
```
score = recency × 0.6 + emotional_weight × 0.4

recency        = exp(-0.15 × (maxDay - chunkDay))
emotional_weight:
  negative → 0.8   (high weight — emotionally significant)
  positive → 0.5
  playful  → 0.4
  neutral  → 0.2
```

**Result on the sister query:**

| Rank | Day | Sentiment | Score |
|------|-----|-----------|-------|
| #1 | 7 | playful | 0.76 |
| #2 | 4 | negative | 0.56 |
| #3 | 2 | positive | 0.42 |

**Contradiction flagged:** Day 2 (positive) ↔ Day 4 (negative)

**Merged answer:** Synthesizes all 3 chunks chronologically, flags the contradiction, resolves it by noting the emotional shift was in Alex, not the relationship.

---

## Part 4 — System Design

See [`SYSTEM_DESIGN.md`](./SYSTEM_DESIGN.md) for the full architecture doc with storage layers, sync rules, and conflict resolution strategy.

---

## Running Locally

```bash
# Clone
git clone https://github.com/Karthaa/kastackL2.git
cd kastackL2/kastack-demo

# Install
npm install

# Run
npm run dev
# → http://localhost:5173
```

**Requirements:** Node.js 18+, npm 9+

---

## Project Structure

```
kastackL2/
├── kastack-demo/              ← React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx            ← All 4 parts in one component
│   │   └── main.jsx
│   ├── package.json
│   └── index.html
│
├── src/                       ← Python backend modules
│   ├── part1/
│   │   └── drift_detector.py  ← Persona drift engine
│   ├── part2/
│   │   └── intent_classifier.py ← TF-IDF + LR model
│   ├── part3/
│   │   └── rag_resolver.py    ← RAG conflict resolver
│   └── part4/
│       └── sync_design.py     ← Architecture notes
│
├── SYSTEM_DESIGN.md           ← On-device storage + sync doc
└── README.md
```

---

## Tech Decisions

**Why Naive Bayes over Logistic Regression for the browser?**
Scikit-learn can't run in a browser. NB is mathematically equivalent for TF-IDF features, trains in milliseconds, and produces well-calibrated probabilities after softmax normalization.

**Why recency × 0.6 + emotional × 0.4 for RAG scoring?**
Recent context is more likely to reflect current user state (hence 60% weight). But emotionally charged chunks carry more signal for personal memory queries — a stressful mention is more memorable than a neutral one (hence 40% weight for emotional).

**Why negative sentiment gets higher emotional weight (0.8)?**
Negative emotional events have higher recall in human memory (negativity bias). For a personal assistant, surfacing high-stress mentions is more useful than neutral ones.

---

## Self-Evaluation

| Criterion | Score | Notes |
|-----------|-------|-------|
| Drift detection correctness | ✅ | Detects 2 drift events with correct triggers |
| Intent classifier (trained, not rules) | ✅ | Real TF-IDF + NB trained in browser |
| RAG relevance & conflict handling | ✅ | Scoring formula + contradiction flagging |
| System design pros/cons | ✅ | CRDT, privacy boundary, GDPR in SYSTEM_DESIGN.md |
| Working end-to-end | ✅ | Single `npm run dev` runs everything |
| Offline (no API calls in classifier) | ✅ | Zero network requests for Parts 1–3 |
| Under 200ms inference | ✅ | ~5–10ms measured |
| Model under 50MB | ✅ | ~50 KB |

---

## Author

Built for KaStack Labs AI/ML Engineer Intern — L2 assessment.
