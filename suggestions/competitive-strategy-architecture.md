# DanexChat competitiveness strategy (speed, memory, accuracy)

This document proposes a practical, implementable strategy to make DanexChat more competitive with larger models while preserving on-device efficiency.

---

## 1) Product strategy: where to compete

DanexChat should not compete by trying to match frontier models on raw open-domain intelligence. It should compete on:

1. **Instant UX on-device**
   - Fast first token
   - Stable latency
   - Offline reliability

2. **High trust for targeted tasks**
   - Fewer hallucinations on supported domains
   - Clear uncertainty behavior outside supported domains
   - Grounded responses when local knowledge exists

3. **Personal context quality**
   - Better memory of user goals/preferences
   - Robust long conversation continuity
   - Privacy-preserving local personalization

This is a strong competitive position for small models: **use architecture and retrieval to close quality gaps**, not brute-force parameter count.

---

## 2) Core system architecture (recommended)

Use a multi-stage pipeline rather than single-pass generation:

1. **Input classifier/router**
   - Classify request: factual / creative / coding / planning / personal-memory query.
   - Route to decoding profile + tools policy.

2. **Context builder**
   - Build prompt from:
     - system policy
     - recent chat turns
     - long-term memory summary
     - retrieved snippets (if available)
   - Enforce token budget with deterministic priority rules.

3. **Retriever (optional but preferred)**
   - Local vector + lexical retrieval from:
     - conversation memory store
     - user notes/docs
     - curated local KB bundles

4. **Generator**
   - Small primary LLM optimized for speed.
   - Optional tiny verifier pass for high-risk/factual answers.

5. **Post-processor**
   - Contradiction checks
   - citation/grounding checks
   - safety/style cleanup

This architecture adds quality without requiring a large base model.

---

## 3) Model architecture options to implement

### Option A: Single small LLM + retrieval + verifier (best near-term)

- **Main model**: compact chat model for generation.
- **Retriever**: local BM25 + embedding index.
- **Verifier model**: tiny NLI/consistency checker or short second-pass prompt.

Why it works:
- Most perceived quality gains come from better context and verification, not just bigger model size.
- Keeps inference memory small while adding factual discipline.

### Option B: Mixture-of-Experts at application layer

Use multiple lightweight specialists instead of one larger model:
- general responder
- coding responder
- planning/structured responder
- safety/consistency checker

A router chooses one or two specialists per turn. This can outperform a single small model on mixed workloads with modest overhead.

### Option C: Distilled dual-model stack

- **Fast draft model** for first response.
- **Refine model** (slightly larger or tuned) for selected cases:
  - long answers
  - factual claims
  - low-confidence signals

Trigger refine only when needed to keep median latency low.

---

## 4) Speed, memory, and accuracy optimization plan

### Speed

1. **Warm start pipeline**
   - Preload tokenizer and run one tiny dry forward pass after app launch.

2. **Adaptive decoding profiles**
   - factual: low temperature, shorter max tokens
   - creative: higher diversity
   - coding: constrained formatting + moderate temperature

3. **Latency-aware routing**
   - Skip expensive verification for low-risk short answers.
   - Enable verification only on high-risk/factual turns.

4. **Streaming prioritization**
   - Emit concise answer first, details second.
   - Improves perceived speed even when total generation time is unchanged.

### Memory

1. **Quantization tiers by device class**
   - low-end: lower-bit model + shorter generation cap
   - mid/high-end: higher quality quantization and larger context windows

2. **KV-cache controls**
   - Keep pinned system and summary tokens.
   - Segment cache for recent turns to reduce recompute.

3. **Context compaction**
   - Replace old turns with structured summaries.
   - Keep raw recent turns only.

4. **On-device indexes with limits**
   - Cap index size and use recency/importance pruning.
   - Maintain deterministic memory budget.

### Accuracy

1. **Grounding-first policy for factual tasks**
   - If retrieval exists, generate from retrieved facts.
   - If not, default to calibrated uncertainty + follow-up questions.

2. **Claim-level validation**
   - Extract high-risk claims and run quick support checks.
   - Rewrite unsupported absolutes to uncertainty language.

3. **Task-specific prompt templates**
   - stable, tested templates for factual/coding/planning tasks.

4. **Continuous offline evaluation**
   - Track metrics:
     - factual precision on benchmark prompts
     - contradiction rate
     - tool-use correctness
     - latency/energy per response

---

## 5) Long-context design (how to make context longer effectively)

Large raw context windows are expensive and often wasteful. Use a hybrid strategy:

### A) Hierarchical memory layers

1. **Working memory (short-term)**
   - last N turns (raw text)

2. **Session summary (mid-term)**
   - compressed running summary:
     - goals
     - known facts
     - decisions
     - unresolved questions

3. **Long-term memory (cross-session)**
   - structured user profile + key durable facts
   - retrieved by relevance, not blindly appended

### B) Retrieval-first long context

- Store chunks with metadata:
  - source type
  - timestamp
  - topic tags
  - confidence
- At generation time:
  - retrieve top-k semantically relevant chunks
  - rerank by recency + confidence + intent match
  - inject only highest value chunks

### C) Token budget policy

Use fixed budget buckets, for example:
- 15% system and behavior policy
- 25% recent turns
- 25% session summary
- 30% retrieved evidence
- 5% response safety margin

If over budget, trim in this order:
1) lower-ranked retrieved chunks
2) oldest recent turns
3) compress summary further
4) never drop core system policy

### D) Attention-sink + anchor strategy

- Keep stable anchor tokens at start:
  - identity
  - style contract
  - task objective
- Preserve latest turn block and summary anchors during trimming.
- Prevent context drift over long sessions.

### E) Memory write policy

Write memory only when information is:
- durable (likely useful later),
- user-specific,
- confidence above threshold.

Avoid storing transient or uncertain claims as durable memory.

---

## 6) Suggested implementation roadmap

### Phase 1: immediate (low risk)
- Add router + decoding profiles (factual/creative/coding).
- Add structured session summary update per turn.
- Add optional contradiction-check pass for long/high-risk outputs.
- Add measured telemetry for latency/tokens/response quality signals.

### Phase 2: quality infrastructure
- Implement local retrieval (hybrid lexical + embedding).
- Add claim extraction and support checks.
- Add cross-session long-term memory store with relevance retrieval.

### Phase 3: advanced competitiveness
- Add draft-then-refine gated pipeline.
- Add specialist responder routing for code/planning/factual tasks.
- Add benchmark suite + regression gates for quality vs latency tradeoffs.

---

## 7) Practical evaluation targets

Set explicit release gates so changes are competitive and efficient:

- **Speed**
  - first-token latency target by device tier
  - p95 end-to-end response latency

- **Memory**
  - peak RAM cap by tier
  - bounded retrieval index size

- **Accuracy**
  - contradiction rate down
  - factual support rate up (when retrieval enabled)
  - user-rated helpfulness/coherence up

If a change improves quality but violates speed/memory thresholds, gate it behind adaptive routing or device-tier controls.

---

## 8) Key principle

To be competitive with larger models on-device:

**Use a smaller model as the language engine, and win with system architecture**—routing, retrieval, memory design, and verification.

That combination provides the best path to strong practical quality while staying efficient in speed and memory.
