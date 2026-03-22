# DanexChat

An Android chat application powered by **SmolLM2-135M-Instruct** running entirely **on-device** using ONNX Runtime. No internet connection required for inference — the AI model runs locally on your Android device.

## Features

- 💬 Chat interface with streaming token-by-token output
- 🤖 **SmolLM2-135M-Instruct** — a compact, capable language model by HuggingFace
- ⚡ Quantized (Q4) ONNX model for fast on-device inference
- 🔐 100% on-device — no data leaves your phone after first download
- 📱 Supports **Android 11–15** (API 30–35)

## Architecture

```
app/
├── SmolLMInference.java   – ONNX Runtime inference engine with KV-cache
├── BPETokenizer.java      – Byte-level BPE tokenizer (loads tokenizer.json)
├── ModelManager.java      – Downloads / caches model files from HuggingFace
├── ChatAdapter.java       – RecyclerView chat bubble adapter
├── MainActivity.java      – Chat UI, download overlay, streaming responses
└── Message.java           – Chat message data class
```

## How It Works

1. **First launch**: the app downloads `model_q4.onnx` (~90 MB) and `tokenizer.json` from HuggingFace into the app's internal storage.
2. **Model load**: ONNX Runtime initialises the session (uses NNAPI acceleration where available).
3. **Chat**: each user message is tokenised with the SmolLM2 ChatML template, run through the ONNX model with greedy decoding, and streamed token-by-token to the UI.

### Model details

| Property | Value |
|---|---|
| Model | `onnx-community/SmolLM2-135M-Instruct` |
| Format | ONNX Q4 quantized |
| Download size | ≈ 90 MB |
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

> **Note**: The first time the app opens it will download the ~90 MB model file. Make sure the device has Wi-Fi access and around 200 MB of free storage.

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| ONNX Runtime Android | 1.20.0 | On-device inference |
| AndroidX AppCompat | 1.7.0 | Activity / theme base |
| Material Components | 1.12.0 | UI widgets |
| ConstraintLayout | 2.2.0 | Layouts |
| RecyclerView | 1.3.2 | Chat message list |

## License

Apache 2.0 — see [LICENSE](LICENSE).