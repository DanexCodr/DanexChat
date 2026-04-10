# DanexChat

An Android chat application powered by **SmolLM2-135M-Instruct** running entirely **on-device** using ONNX Runtime. No internet connection required for inference — the AI model runs locally on your Android device.

## Features

- 💬 Chat interface with streaming token-by-token output
- 🤖 **SmolLM2-135M-Instruct** — a compact, capable language model by HuggingFace
- ⚡ Quantized (Q4) ONNX model for fast on-device inference
- 🔐 100% on-device — no data leaves your phone
- 🧠 Single-conversation flow with lightweight ambiguity/topic handling
- 🔍 **BERT-tiny** semantic encoder for intent routing, factual retrieval, and response caching
- 📖 Bundled WordNet-based factual dictionary for grounded definitions and facts
- 🏋️ Fine-tunable: retraining pipeline (pruning + SFT) available under `training/`
- 📱 Supports **Android 11–15** (API 30–35)

## Architecture

```
app/
├── MainActivity.java         – Single chat UI, streaming flow, context heuristics
├── SmolLMInference.java      – ONNX Runtime inference engine + decoding + KV-cache path
├── BPETokenizer.java         – Byte-level BPE tokenizer (loads tokenizer.json)
├── BertTinyEncoder.java      – BERT-tiny ONNX encoder for semantic embeddings
├── BertTinyTokenizer.java    – WordPiece tokenizer for BERT-tiny
├── IntentRouter.java         – Routes queries to factual retrieval or generative path
├── SemanticResponseCache.java – Caches recent responses by BERT embedding similarity
├── FactualDictionary.java    – Loads and looks up the bundled WordNet factual dictionary
├── ModelManager.java         – Ensures bundled model/tokenizer/assets are in app storage
├── ChatAdapter.java          – RecyclerView chat bubble adapter
└── Message.java              – Chat message data class
```

### Current AI architecture (runtime flow)

1. **Startup + asset readiness**
   - `MainActivity` calls `ModelManager.isReady()` on a background executor.
   - `ModelManager` verifies `assets/smollm2/model_q4.onnx`, `tokenizer.json`, `wordnet.json`, and the BERT-tiny files (`bert_tiny/bert_tiny.onnx`, `bert_tiny/bert_vocab.txt`) are all copied to internal storage and pass minimum size checks.

2. **Inference engine initialization**
   - `SmolLMInference` creates one ONNX Runtime session and tokenizer instance.
   - It discovers whether the model export includes KV-cache tensors and whether `position_ids` are required.
   - Generation options use fixed defaults (temperature/top-p/max tokens), with deterministic temperature set to `0`.
   - A bundled WordNet-based factual dictionary (`assets/smollm2/wordnet.json`) is loaded at startup and injected into prompts as local grounding hints.
   - `BertTinyEncoder` loads a separate BERT-tiny ONNX session for semantic encoding used by routing and caching.

3. **Prompt construction + generation**
   - `IntentRouter` uses BERT-tiny embeddings to classify the query and route it to factual retrieval or the generative path.
   - Definition-style queries short-circuit to exact local dictionary entries from `FactualDictionary` before model generation when an exact subject match exists.
   - Chat messages are transformed into a ChatML-style prompt with a fixed DanexChat system instruction and any matched dictionary facts as a grounding block.
   - Prompt tags (`$date`, `$time`, `$datetime`, `$route`) are resolved at runtime before generation.
   - During generation, decoding applies top-p filtering with deterministic token choice (temperature `0`) and streams token pieces back to UI incrementally.
   - `SemanticResponseCache` checks BERT-embedding similarity of recent queries; sufficiently similar queries reuse the cached response.
   - A fallback path retries with `position_ids` if the runtime reports a missing input.

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
| Main model | `onnx-community/SmolLM2-135M-Instruct` |
| Format | ONNX Q4 quantized |
| Model size in APK/assets | ≈ 90 MB |
| Semantic encoder | `prajjwal1/bert-tiny` (exported to ONNX) |
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

> **Note**: This project expects all model assets to be pre-bundled before APK build:
> - `app/src/main/assets/smollm2/model_q4.onnx`
> - `app/src/main/assets/smollm2/tokenizer.json`
> - `app/src/main/assets/smollm2/wordnet.json`
> - `app/src/main/assets/bert_tiny/bert_tiny.onnx`
> - `app/src/main/assets/bert_tiny/bert_vocab.txt`
>
> You can generate the BERT-tiny files with:
> ```bash
> python training/export_bert_tiny.py --output-dir /tmp/bert_tiny_assets
> mkdir -p app/src/main/assets/bert_tiny
> cp /tmp/bert_tiny_assets/bert_tiny.onnx app/src/main/assets/bert_tiny/bert_tiny.onnx
> cp /tmp/bert_tiny_assets/bert_vocab.txt app/src/main/assets/bert_tiny/bert_vocab.txt
> # if export produced external data sidecar, copy it too:
> [ -f /tmp/bert_tiny_assets/bert_tiny.onnx.data ] && cp /tmp/bert_tiny_assets/bert_tiny.onnx.data app/src/main/assets/bert_tiny/bert_tiny.onnx.data
> ```

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| ONNX Runtime Android | 1.20.0 | On-device inference (SmolLM2 + BERT-tiny) |
| AndroidX AppCompat | 1.7.0 | Activity / theme base |
| Material Components | 1.12.0 | UI widgets |
| ConstraintLayout | 2.2.0 | Layouts |
| RecyclerView | 1.3.2 | Chat message list |
| org.json | 20240303 | Tokenizer JSON parsing |

## Training pipeline

A Python retraining pipeline is available under `training/` for custom fine-tuning or pruning of the chat model.

| Script | Purpose |
|---|---|
| `training/prepare_chat_dataset.py` | Builds a SFT-ready chat dataset |
| `training/prune_model.py` | Magnitude-prunes the SmolLM2 model |
| `training/finetune_chat_sft.py` | SFT fine-tunes the pruned model |
| `training/export_bert_tiny.py` | Exports `prajjwal1/bert-tiny` to ONNX |
| `training/run_pipeline.sh` | Orchestrates the full pipeline end-to-end |

See [`training/README.md`](training/README.md) for full usage instructions. A GitHub Actions workflow (`.github/workflows/training.yml`) allows the pipeline to be triggered on demand via `workflow_dispatch`.

## Additional documentation

- [`CHANGELOG.md`](CHANGELOG.md) — versioned history of all significant changes.
- [`suggestions/model-inference.md`](suggestions/model-inference.md) — practical roadmap for improving factuality, coherence, and logic in generation.
- [`suggestions/competitive-strategy-architecture.md`](suggestions/competitive-strategy-architecture.md) — longer-term architecture and competitiveness strategy.
- [`Limitations.md`](Limitations.md) — detailed statement of current architectural/behavioral limits.

## License

Apache 2.0 — see [LICENSE](LICENSE).
