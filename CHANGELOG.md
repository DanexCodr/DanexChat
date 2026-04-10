# Changelog

All notable changes to DanexChat are documented here.
Entries are grouped by logical version milestones and ordered newest-first within each section.

---

## [0.9.0] – 2026-04-02

### Added
- **BERT-tiny semantic layer** (`BertTinyEncoder`, `BertTinyTokenizer`): pre-bundled `prajjwal1/bert-tiny` ONNX encoder for intent routing, factual retrieval, and response caching. (PR #31)
- `IntentRouter`: routes queries to factual retrieval or generative path based on BERT embeddings. (PR #31)
- `SemanticResponseCache`: caches recent responses keyed by BERT embedding cosine similarity. (PR #31)
- Gradle `verifyBundledAiAssets` task enforces that `bert_tiny/bert_tiny.onnx` and `bert_tiny/bert_vocab.txt` are present before any build. (PR #33)
- CI: `build-apk.yml` now generates BERT-tiny assets via `training/export_bert_tiny.py` and caches them between runs. (PR #32, #34)
- Optional `bert_tiny.onnx.data` external-data sidecar support in `ModelManager`, Gradle asset check, and CI packaging. (PR #39)

### Fixed
- Removed vulnerable `onnxruntime` pip dependency from BERT export path. (PR #31)
- CI BERT export now installs `onnxscript` to satisfy `torch.onnx` requirement in newer torch. (PR #38)
- CI BERT export pins `transformers<5` and `tokenizers<0.20` to avoid tokenizer backend `ValueError`. (PR #37)
- CI BERT export installs `sentencepiece` for `AutoTokenizer` conversion. (PR #36)
- CI BERT export installs `protobuf` alongside `torch` and `transformers`. (PR #35)
- CI workflow uses canonical `prajjwal1/bert-tiny` generation path via `export_bert_tiny.py`. (PR #34)
- CI workflow fails fast with descriptive messages when BERT assets are missing or undersized. (PR #33)

---

## [0.8.0] – 2026-04-01

### Added
- Bundled WordNet-based factual dictionary (`assets/smollm2/wordnet.json`) replaces the previous remote dictionary download. (PR #29)
- `FactualDictionary` now short-circuits definition-style queries to exact local dictionary entries before model generation. (PR #30)
- Comprehensive competitiveness strategy and architecture blueprint added to `suggestions/competitive-strategy-architecture.md`. (PR #28)

### Changed
- Decoding is now fully deterministic: temperature fixed to `0`, repetition-penalty output enforcement removed. (PR #27)
- `ModelManager` readiness check now includes `wordnet.json` alongside model and tokenizer. (PR #29)

---

## [0.7.0] – 2026-03-24

### Added
- Runtime prompt tags (`$date`, `$time`, `$datetime`, `$route`) resolved before generation. (PR #26)
- Factual dictionary grounding: matched dictionary facts are injected as a system block in the prompt. (PR #26)
- Lightweight query routing (`IntentRouter` precursor) differentiates factual vs. generative queries. (PR #26)

---

## [0.6.0] – 2026-03-24

### Added
- Python retraining pipeline under `training/`: `prepare_chat_dataset.py`, `prune_model.py`, `finetune_chat_sft.py`, `run_pipeline.sh`. (PR #24)
- GitHub Actions workflow `.github/workflows/training.yml` for on-demand pipeline execution via `workflow_dispatch`. (PR #25)
- `training/export_bert_tiny.py` for exporting `prajjwal1/bert-tiny` to ONNX. (PR #31)

---

## [0.5.0] – 2026-03-24

### Added
- Hard enforcement of canonical DanexChat identity and creator attribution in final responses. (PR #21)
- Direct self-identity questions (e.g., "who are you", "what's your name") now return only the DanexChat identity string. (PR #23)

### Removed
- Settings feature removed; all inference defaults (temperature, top-p, max new tokens) are now hardcoded. (PR #22)

---

## [0.4.0] – 2026-03-23

### Added
- Hierarchical context summarization: recent turns kept in full, older turns compressed into a rolling summary, oldest compressed into an archived summary. (PR #19)
- Attention-sink context trimming integrated with hierarchical summaries. (PR #20)

---

## [0.3.0] – 2026-03-23

### Added
- `suggestions/model-inference.md`: inference improvement roadmap documentation. (PR #18)

### Changed
- Simplified chat architecture to a **single-session flow**; ViewPager multi-session UI removed. (PR #18)
- Auto-scroll and scroll-to-bottom helper controls removed. (PR #18)
- Assistant identity prompt corrected; ARS confidence weighting removed. (PR #17)

### Fixed
- Session pager visibility/clipping issues resolved. (PR #16)
- Real settings controls implemented and rigid identity priming removed. (PR #16)

---

## [0.2.0] – 2026-03-23

### Added
- Multi-session chat UX with tabbed settings and generation/scroll behaviour updates for DanexChat identity. (PR #15)
- Confidence-gated ARS disambiguation with clarification fallback. (PR #14)
- ARS pre-processing for ambiguous definition prompts in chat flow. (PR #13)

### Changed
- Explicit context-switch handling to improve chat topic coherence. (PR #12)

### Fixed
- Repetition controls added to token selection for improved decoding stability. (PR #11)
- SmolLM pre-tokenisation aligned with GPT-2 byte-level chunking to reduce repetitive malformed replies. (PR #10)

---

## [0.1.0] – 2026-03-23 (initial release)

### Added
- On-device chat application powered by `SmolLM2-135M-Instruct` via ONNX Runtime.
- Byte-level BPE tokenizer (`BPETokenizer`) loading `tokenizer.json`.
- Streaming token-by-token output to a `RecyclerView`-based chat UI.
- `ModelManager` for bundled model/tokenizer asset readiness verification.
- Support for Android 11–15 (API 30–35).
- Apache 2.0 license.
