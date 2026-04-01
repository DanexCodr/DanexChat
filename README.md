# DanexChat

An Android chat application powered by **SmolLM2-135M-Instruct** running entirely **on-device** using ONNX Runtime. No internet connection required for inference — the AI model runs locally on your Android device.

## Features

- 💬 Chat interface with streaming token-by-token output
- 🤖 **SmolLM2-135M-Instruct** — a compact, capable language model by HuggingFace
- ⚡ Quantized (Q4) ONNX model for fast on-device inference
- 🔐 100% on-device — no data leaves your phone
- 🧠 Single-conversation flow with lightweight ambiguity/topic handling
- 📱 Supports **Android 11–15** (API 30–35)

## Architecture

```
app/
├── MainActivity.java         – Single chat UI, streaming flow, context heuristics
├── SmolLMInference.java      – ONNX Runtime inference engine + decoding + KV-cache path
├── BPETokenizer.java         – Byte-level BPE tokenizer (loads tokenizer.json)
├── ModelManager.java         – Ensures bundled model/tokenizer are available in app storage
├── ChatAdapter.java          – RecyclerView chat bubble adapter
└── Message.java              – Chat message data class
```

### Current AI architecture (runtime flow)

1. **Startup + asset readiness**
   - `MainActivity` calls `ModelManager.isReady()` on a background executor.
   - `ModelManager` verifies `assets/smollm2/model_q4.onnx` and `tokenizer.json` are copied to internal storage and pass minimum size checks.

2. **Inference engine initialization**
   - `SmolLMInference` creates one ONNX Runtime session and tokenizer instance.
   - It discovers whether the model export includes KV-cache tensors and whether `position_ids` are required.
   - Generation options use fixed defaults (temperature/top-p/max tokens), with deterministic temperature set to `0`.

3. **Prompt construction + generation**
   - Chat messages are transformed into a ChatML-style prompt with a fixed DanexChat system instruction.
   - During generation, decoding applies repetition controls and top-p filtering with deterministic token choice (temperature `0`) and streams token pieces back to UI incrementally.
   - A fallback path retries with `position_ids` if the runtime reports missing input.

4. **Conversation management**
    - The app uses a single in-memory conversation history.
   - Long chats are compacted with hierarchical summarization:
     - recent exchanges are kept in full form,
     - older turns are compressed into a rolling summary,
     - older summary slices are condensed into a high-level archived summary.
    - Lightweight heuristics detect topic shifts for definition-style prompts and reset model-side context when overlap is very low.
    - Ambiguous references (e.g., "it", "this") are rewritten toward the latest concrete subject where possible.

### Model details

| Property | Value |
|---|---|
| Model | `onnx-community/SmolLM2-135M-Instruct` |
| Format | ONNX Q4 quantized |
| Model size in APK/assets | ≈ 90 MB |
| Inference engine | ONNX Runtime Android 1.20.0 |
| Minimum Android | API 30 (Android 11) |
| Target Android | API 35 (Android 15) |

## Building

### Prerequisites

- Android Studio Hedgehog (2023.1.1) or newer
- Android SDK API 35
- JDK 17
- Internet access (to download Gradle dependencies from Google Maven)

### Steps

```bash
git clone https://github.com/DanexCodr/DanexChat.git
cd DanexChat
./gradlew assembleDebug
```

The resulting APK will be at `app/build/outputs/apk/debug/app-debug.apk`.

### Install on device

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

> **Note**: This project expects `app/src/main/assets/smollm2/model_q4.onnx` and `app/src/main/assets/smollm2/tokenizer.json` to be present when building.

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| ONNX Runtime Android | 1.20.0 | On-device inference |
| AndroidX AppCompat | 1.7.0 | Activity / theme base |
| Material Components | 1.12.0 | UI widgets |
| ConstraintLayout | 2.2.0 | Layouts |
| RecyclerView | 1.3.2 | Chat message list |

## Additional documentation

- [`suggestions/model-inference.md`](suggestions/model-inference.md) — practical roadmap for improving factuality, coherence, and logic in generation.
- [`Limitations.md`](Limitations.md) — detailed statement of current architectural/behavioral limits.

## License

Apache 2.0 — see [LICENSE](LICENSE).
