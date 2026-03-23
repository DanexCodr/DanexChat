# DanexChat Current Limitations

This document describes what DanexChat **cannot reliably handle yet** in its current architecture.

## 1) Knowledge and factuality limitations

1. **No live internet retrieval**
   - The model runs fully on-device and does not fetch live web data.
   - It cannot verify breaking news, recent events, or changing facts in real time.

2. **No authoritative source grounding**
   - Responses are generated from model parameters and current conversation context only.
   - There is no built-in citation or source-backed verification pipeline.

3. **Hallucination risk remains**
   - For uncertain or niche topics, the model may produce plausible but incorrect statements.
   - This risk increases for precise data (dates, statistics, legal/medical specifics).

## 2) Reasoning and coherence limitations

1. **Small model constraints (SmolLM2-135M)**
   - Compact model size is fast and efficient, but limits deep multi-step reasoning.
   - Complex planning, advanced logic chains, and domain-heavy analysis can degrade.

2. **Single-pass generation**
   - Responses are streamed token-by-token without a mandatory second verification/refinement pass.
   - Contradictions or weak logic may appear in long answers.

3. **Context window constraints**
   - The effective prompt is bounded by max context length.
   - DanexChat now uses hierarchical summarization to keep recent turns intact while compressing older turns into summary layers.
   - Compression preserves key facts better than raw truncation, but some fine-grained details can still be lost in very long chats.

## 3) Conversation and memory limitations

1. **No multi-session UI**
   - The app currently runs as a single chat stream in one main conversation view.

2. **No durable chat persistence**
   - Conversation state is in memory for the running process.
   - Restarting the app clears prior session context.

3. **Topic shift heuristics**
   - Topic switching and ambiguity rewriting are heuristic-based and may occasionally over/under-reset context.

## 4) UX and control limitations

1. **No auto-scroll helper controls**
   - Automatic scroll-follow behavior and manual scroll-to-bottom helper controls are intentionally removed.
   - Long responses may require manual navigation in the message list.

2. **Limited generation controls**
   - Settings currently expose core decoding knobs (temperature, top-p, max new tokens) only.
   - No per-prompt presets (e.g., factual vs creative) yet.

## 5) Platform/runtime limitations

1. **Model packaging footprint**
   - The ONNX model asset is large, increasing APK/app storage usage.

2. **Device-dependent performance**
   - Inference latency varies significantly by CPU, thermal conditions, and available memory.

3. **Potential startup delay on first load**
   - Initial model/session setup and tokenizer load can be noticeable on slower devices.

4. **Android version floor**
   - Current app targets modern Android APIs and does not support older API levels below project minimum.

## 6) Tooling/testing limitations

1. **No dedicated unit/instrumentation test suite**
   - Validation is currently build-centric (assemble), with limited automated behavioral coverage.

2. **Environment-dependent build resolution**
   - Build success can depend on external Gradle/plugin repository availability.

---

## Practical guidance for users

- Treat responses as assistance, not guaranteed truth.
- For high-stakes decisions (medical, legal, financial, safety), verify with trusted experts/sources.
- Ask focused follow-up questions when an answer appears uncertain or too broad.
