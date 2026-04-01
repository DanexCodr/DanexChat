# Model inference improvement suggestions

This document focuses on practical ideas to make DanexChat outputs more factual, coherent, and logically consistent while preserving fast on-device inference.

## Immediate improvements (low-risk, high-impact)

1. **Deterministic factual mode toggle**
   - Add a "Factual mode" preset that sets:
      - temperature: `0`
      - top-p: `0.7-0.85`
      - max_new_tokens: context-dependent (short for direct factual queries)
   - Why: reduces hallucinated drift and keeps reasoning tighter for knowledge questions.

2. **Prompt-level answer structure hints**
   - Update system prompt to enforce:
     - answer first in 1-2 lines,
     - then bullet evidence/assumptions,
     - then uncertainty note when confidence is low.
   - Why: makes reasoning chain visible without exposing hidden chain-of-thought.

3. **Fast contradiction guardrail**
   - Before finalizing, run a lightweight second pass prompt:
     - "Check the previous answer for contradictions and unsupported claims. Rewrite only if needed."
   - Only execute when response length exceeds a threshold or contains certainty words ("always", "never", exact dates, etc.).
   - Why: catches self-contradictions with minimal extra compute.

4. **Uncertainty calibration instruction**
   - Require model to use calibrated language for uncertain claims:
     - "likely", "possibly", "I may be mistaken", and ask a clarifying follow-up.
   - Why: improves factual honesty in offline/no-retrieval mode.

5. **Hard-stop cleanup**
   - Post-process response end for incomplete sentences and truncated bullet items when user presses Stop or max tokens hit.
   - Why: better coherence and readability.

## Bug and gap fixes worth doing next

1. **Session persistence gap**
   - Current chat/session history is in-memory only.
   - Add optional local persistence (encrypted SharedPreferences/Room) for continuity after app restart.

2. **No source-grounding gap**
   - Without retrieval, model cannot verify claims against trusted sources.
   - Add optional local document grounding (small local corpus + BM25/embedding retrieval).

3. **Single-pass generation limitation**
   - Current generation path is single-pass streaming only.
   - Introduce optional "draft then refine" for long answers (2 short passes).

4. **No query-type routing**
   - Factual, creative, and coding prompts currently share one decoding profile.
   - Add rule-based routing: factual profile vs creative profile.

## Impactful ideas (medium effort)

1. **On-device RAG lite**
   - Index user-provided notes/docs locally.
   - Retrieve top-k snippets and inject with strict citation template.
   - Require claims to map to retrieved snippets when available.

2. **Claim extraction + verification loop**
   - Step A: extract atomic claims from draft answer.
   - Step B: verify each claim against retrieved snippets or mark unverified.
   - Step C: regenerate final answer with verified-only factual statements.

3. **Conversation state summarization**
   - When context is long, maintain rolling structured summary:
     - user goals,
     - accepted facts,
     - unresolved questions.
   - Feed summary + recent turns instead of raw long history.

4. **Safety for arithmetic/logical tasks**
   - Add deterministic micro-tools (local calculator/parser) for arithmetic/date logic.
   - Let model call tool output for exact computations.

## Inference/runtime optimizations that help quality indirectly

1. **Warmup run at model load**
   - Perform one tiny dummy forward pass immediately after session creation.
   - Why: reduces first-token latency spikes and gives smoother UX.

2. **Thread tuning by device class**
   - Current thread count is static cap.
   - Tune intra/inter-op threads by CPU cores and thermal state.

3. **Adaptive max token limit**
   - Short factual queries should default to shorter generation lengths.
   - Helps reduce rambling and accidental contradictions.

## Suggested roadmap

### Phase 1 (quick wins)
- Add factual mode preset
- Improve prompt format for answer/evidence/uncertainty
- Add response contradiction check for long outputs
- Add warmup run after model load

### Phase 2 (quality infrastructure)
- Add conversation summary memory
- Add query-type decoding routing
- Add local persistence for user context

### Phase 3 (major factual gains)
- Add local RAG-lite with snippet injection
- Add claim extraction/verification pipeline

## Expected outcomes

- More direct and coherent responses for factual queries
- Lower hallucination rate in unsupported topics
- Better logical consistency across longer answers
- Improved first-token latency experience
