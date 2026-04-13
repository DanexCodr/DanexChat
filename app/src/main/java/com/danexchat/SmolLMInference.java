package com.danexchat;

import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.ThreadLocalRandom;
import java.util.regex.Pattern;

/**
 * On-device inference engine for SmolLM2-135M-Instruct using ONNX Runtime.
 *
 * The model is exported by HuggingFace optimum (onnx-community/SmolLM2-135M-Instruct)
 * and uses merged KV-cache tensors:
 *   Inputs : input_ids, attention_mask,
 *            past_key_values.{i}.key / past_key_values.{i}.value  (i = 0..N-1)
 *   Outputs: logits,
 *            present.{i}.key / present.{i}.value                   (i = 0..N-1)
 *
 * Must be called from a background thread.
 */
public class SmolLMInference {

    private static final String TAG = "SmolLMInference";
    private static final Pattern MISSING_POSITION_IDS_PATTERN =
            Pattern.compile("\\bmissing input:\\s*position_ids\\b", Pattern.CASE_INSENSITIVE);

    // SmolLM2-135M architecture
    private static final int NUM_LAYERS   = 30;
    private static final int NUM_KV_HEADS = 3;
    private static final int HEAD_DIM     = 64;

    // Generation limits
    private static final int MAX_NEW_TOKENS  = 512;
    private static final int MIN_NEW_TOKENS  = 32;
    private static final int MAX_CONTEXT_LEN = 2048;
    private static final int ATTENTION_SINK_TOKEN_COUNT = 128;
    private static final int ATTENTION_SINK_MAX_CONTEXT_DIVISOR = 2;
    private static final float REPETITION_PENALTY = 1.15f;
    private static final int REPETITION_WINDOW = 128;
    private static final int NO_REPEAT_NGRAM_SIZE = 3;
    private static final float MIN_TEMPERATURE = 0.0f;
    private static final float MAX_TEMPERATURE = 2.0f;
    private static final float MIN_TOP_P = 0.05f;
    private static final float MAX_TOP_P = 1.0f;
    private static final float FLOAT_EPSILON = 0.001f;
    private static final float DEFAULT_TEMPERATURE = 0.0f;
    private static final float DEFAULT_TOP_P = 0.9f;
    private static final int DEFAULT_MAX_NEW_TOKENS = 256;
    private static final int MAX_DICTIONARY_FACTS = 3;
    /**
     * Placeholder token that the model is instructed to emit where the dictionary
     * definition should be inserted.  Post-processing replaces this with the verbatim
     * definition text before the response is delivered to the caller.
     */
    private static final String DICT_DEFINITION_PLACEHOLDER = "<<DEFINITION>>";
    // Dictionary template turns should stay short so they complete quickly and avoid long stalls.
    private static final int DICTIONARY_MAX_NEW_TOKENS = 96;
    // Intro paraphrase is intentionally constrained to keep fallback assembly concise and stable.
    private static final int MAX_INTRO_CHARS = 180;
    private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");
    private static final String DICTIONARY_CLOSING_PHRASE = "Do you want to know more about ";
    private static final int HISTORY_RELEVANCE_MIN_SIZE = 8;
    private static final int HISTORY_RELEVANCE_RECENT_KEEP = 4;
    private static final int HISTORY_RELEVANCE_MAX_OLDER = 6;
    private static final float HISTORY_RELEVANCE_MIN_SCORE = 0.15f;
    private static final String TAG_DATE = "$date";
    private static final String TAG_TIME = "$time";
    private static final String TAG_DATETIME = "$datetime";
    private static final String DEFAULT_ROUTE = "general";
    private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd");
    private static final DateTimeFormatter TIME_FORMAT = DateTimeFormatter.ofPattern("HH:mm:ss");
    private static final DateTimeFormatter DATETIME_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final OrtEnvironment env;
    private final OrtSession session;
    private final BPETokenizer tokenizer;
    private final FactualDictionary factualDictionary;
    // Optional BERT-tiny encoder for semantic dictionary lookup; may be null.
    private final BertTinyEncoder bertEncoder;
    private final SemanticResponseCache semanticResponseCache;

    // KV-cache tensor name lists (populated at load time)
    private final List<String> pastKeyNames = new ArrayList<>();
    private final List<String> pastValNames = new ArrayList<>();
    private final List<String> presKeyNames = new ArrayList<>();
    private final List<String> presValNames = new ArrayList<>();
    private boolean hasKVCache = false;
    private boolean requiresPositionIds = false;
    private final AtomicBoolean stopRequested = new AtomicBoolean(false);
    private GenerationOptions generationOptions = GenerationOptions.defaults();

    // -----------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------

    public SmolLMInference(File modelFile, File tokenizerFile)
            throws OrtException, IOException, org.json.JSONException {
        this(modelFile, tokenizerFile, null, null);
    }

    public SmolLMInference(File modelFile, File tokenizerFile, File dictionaryFile)
            throws OrtException, IOException, org.json.JSONException {
        this(modelFile, tokenizerFile, dictionaryFile, null);
    }

    /**
     * Full constructor that accepts an optional {@link BertTinyEncoder} for
     * semantic dictionary fact retrieval. When {@code bertEncoder} is non-null,
     * {@link FactualDictionary#findSemanticFacts} is used instead of the
     * keyword-only {@link FactualDictionary#findRelevantFacts}.
     *
     * <p>Note: this class does not own {@code bertEncoder} and will not close it.
     */
    public SmolLMInference(File modelFile, File tokenizerFile, File dictionaryFile,
                           BertTinyEncoder bertEncoder)
            throws OrtException, IOException, org.json.JSONException {

        env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(
                Math.min(4, Runtime.getRuntime().availableProcessors()));

        session = env.createSession(modelFile.getAbsolutePath(), opts);
        tokenizer = new BPETokenizer(tokenizerFile);
        factualDictionary = dictionaryFile != null && dictionaryFile.exists()
                ? new FactualDictionary(dictionaryFile)
                : null;
        this.bertEncoder = bertEncoder;
        this.semanticResponseCache = bertEncoder != null ? new SemanticResponseCache(bertEncoder) : null;
        discoverKVNames();
        Log.i(TAG, "Loaded SmolLM2. KV-cache layers: " + pastKeyNames.size()
                + ", semantic lookup: " + (bertEncoder != null));
    }

    private void discoverKVNames() throws OrtException {
        Set<String> inputNames  = session.getInputNames();
        Set<String> outputNames = session.getOutputNames();
        requiresPositionIds = inputNames.contains("position_ids");

        for (int i = 0; i < NUM_LAYERS; i++) {
            String pk = "past_key_values." + i + ".key";
            String pv = "past_key_values." + i + ".value";
            if (inputNames.contains(pk)) {
                pastKeyNames.add(pk);
                pastValNames.add(pv);
            } else {
                break;
            }
        }
        for (int i = 0; i < pastKeyNames.size(); i++) {
            presKeyNames.add("present." + i + ".key");
            presValNames.add("present." + i + ".value");
        }
        hasKVCache = !pastKeyNames.isEmpty();
    }

    // -----------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------

    /** Callback for streaming token-by-token output to the UI. */
    public interface StreamCallback {
        void onToken(String tokenText);
        void onComplete(String fullResponse);
        void onError(Exception e);
    }

    static final class GenerationOptions {
        final float temperature;
        final float topP;
        final int maxNewTokens;

        GenerationOptions(float temperature, float topP, int maxNewTokens) {
            this.temperature = clamp(temperature, MIN_TEMPERATURE, MAX_TEMPERATURE);
            this.topP = clamp(topP, MIN_TOP_P, MAX_TOP_P);
            this.maxNewTokens = Math.max(MIN_NEW_TOKENS, Math.min(MAX_NEW_TOKENS, maxNewTokens));
        }

        static GenerationOptions defaults() {
            return new GenerationOptions(
                    DEFAULT_TEMPERATURE,
                    DEFAULT_TOP_P,
                    DEFAULT_MAX_NEW_TOKENS
            );
        }
    }

    /** Build a ChatML-format DanexChat prompt from conversation history. */
    public String buildPrompt(List<Message> history, String summary, String archivedSummary) {
        return buildPrompt(history, summary, archivedSummary, null);
    }

    /** Build a ChatML-format DanexChat prompt from conversation history. */
    public String buildPrompt(
            List<Message> history,
            String summary,
            String archivedSummary,
            Map<String, String> promptTags
    ) {
        Map<String, String> effectiveTags = buildRealtimeTags(promptTags);
        StringBuilder sb = new StringBuilder();
        sb.append("<|im_start|>system\n")
           .append("You are a helpful on-device AI assistant.\n")
            .append("Always answer the user's latest message directly.\n")
            .append("Use earlier messages only when they are relevant to the current request.\n")
            .append("If the user switches topics, switch context immediately and do not continue the old topic.\n")
            .append("If a request is ambiguous, pick the best-supported interpretation and continue directly.\n")
            .append("Keep answers clear, factual, and concise.\n")
            .append("If local factual hints are provided, treat them as the primary source of truth.\n")
            .append("Do not invent facts that are not supported by local factual hints or clear user context.\n")
            .append("If you are unsure, explicitly say you are not fully sure instead of guessing.\n")
            .append("Routing mode: $route.\n")
            .append("Current date is $date, current time is $time, and current datetime is $datetime.\n")
            .append("<|im_end|>\n");
        appendSummaryBlock(sb, "Archived conversation summary", archivedSummary);
        appendSummaryBlock(sb, "Conversation summary", summary);
        appendSummaryBlock(sb, "Factual dictionary hints", buildDictionaryFacts(history));
        for (Message msg : history) {
            sb.append(msg.isUser() ? "<|im_start|>user\n" : "<|im_start|>assistant\n")
              .append(msg.getContent()).append('\n')
              .append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return applyPromptTags(sb.toString(), effectiveTags);
    }

    /**
     * Generate a response for the given conversation history.
     * Must be called on a background thread.
     */
    public synchronized void generate(List<Message> history, StreamCallback callback) {
        generate(history, "", "", callback);
    }

    public synchronized void generate(
            List<Message> history,
            String summary,
            String archivedSummary,
            StreamCallback callback
    ) {
        generate(history, summary, archivedSummary, null, callback);
    }

    public synchronized void generate(
            List<Message> history,
            String summary,
            String archivedSummary,
            Map<String, String> promptTags,
            StreamCallback callback
    ) {
        stopRequested.set(false);
        try {
            generateInternal(history, summary, archivedSummary, promptTags, callback);
        } catch (Exception firstError) {
            if (!requiresPositionIds && isMissingPositionIdsError(firstError)) {
                Log.w(TAG, "Retrying generation with position_ids enabled after runtime error", firstError);
                requiresPositionIds = true;
                try {
                    generateInternal(history, summary, archivedSummary, promptTags, callback);
                    return;
                } catch (Exception retryError) {
                    Log.e(TAG, "Generation retry failed", retryError);
                    callback.onError(retryError);
                    return;
                }
            }
            Log.e(TAG, "Generation error", firstError);
            callback.onError(firstError);
        }
    }

    public synchronized void updateGenerationOptions(GenerationOptions options) {
        if (options != null) {
            generationOptions = options;
        }
    }

    public synchronized void clearResponseCache() {
        if (semanticResponseCache != null) {
            semanticResponseCache.clear();
        }
    }

    private void generateInternal(
            List<Message> history,
            String summary,
            String archivedSummary,
            Map<String, String> promptTags,
            StreamCallback callback
    ) throws Exception {
        String lastUserText = extractLastUserText(history);

        // Check for an exact dictionary match first.  When found, the model is run with
        // template-enforced instructions (without the definition) so it can freely
        // rephrase the beginning/ending while the verbatim definition and topic are
        // injected during post-processing.
        if (factualDictionary != null && lastUserText != null) {
            FactualDictionary.DictionaryMatch dictMatch =
                    factualDictionary.findExactMatch(lastUserText);
            if (dictMatch != null) {
                generateDictionaryResponse(
                        dictMatch, history, summary, archivedSummary, promptTags, callback);
                return;
            }
        }

        if (semanticResponseCache != null && lastUserText != null) {
            String cached = semanticResponseCache.find(lastUserText);
            if (cached != null) {
                callback.onComplete(cached);
                return;
            }
        }
        List<Message> effectiveHistory = selectRelevantHistory(history, lastUserText);
        String prompt = buildPrompt(effectiveHistory, summary, archivedSummary, promptTags);
        long[] promptIds = tokenizer.encodeWithSpecialTokens(prompt);

        // Trim to max context with an attention sink prefix + rolling recent window.
        promptIds = trimPromptWithAttentionSink(promptIds);

        StringBuilder responseBuilder = new StringBuilder();
        StreamCallback wrappedCallback = new StreamCallback() {
            @Override
            public void onToken(String tokenText) {
                callback.onToken(tokenText);
            }

            @Override
            public void onComplete(String fullResponse) {
                if (shouldCacheResponse(lastUserText, fullResponse)) {
                    semanticResponseCache.put(lastUserText, fullResponse);
                }
                callback.onComplete(fullResponse);
            }

            @Override
            public void onError(Exception e) {
                callback.onError(e);
            }
        };
        if (hasKVCache) {
            generateWithKVCache(promptIds, responseBuilder, wrappedCallback);
        } else {
            generateNoKVCache(promptIds, responseBuilder, wrappedCallback);
        }
    }

    private static String extractLastUserText(List<Message> history) {
        if (history == null) return null;
        for (int i = history.size() - 1; i >= 0; i--) {
            Message msg = history.get(i);
            if (msg.isUser()) {
                return msg.getContent();
            }
        }
        return null;
    }

    private List<Message> selectRelevantHistory(List<Message> history, String lastUserText) {
        if (bertEncoder == null
                || history == null
                || history.size() <= HISTORY_RELEVANCE_MIN_SIZE
                || lastUserText == null
                || lastUserText.trim().isEmpty()) {
            return history;
        }
        float[] queryEmbedding = bertEncoder.encode(lastUserText);
        if (queryEmbedding == null) {
            return history;
        }

        int total = history.size();
        int recentStart = Math.max(0, total - HISTORY_RELEVANCE_RECENT_KEEP);
        int olderCount = recentStart;
        if (olderCount <= 0) {
            return history;
        }

        float[] scores = new float[olderCount];
        for (int i = 0; i < olderCount; i++) {
            float[] messageEmbedding = bertEncoder.encode(history.get(i).getContent());
            scores[i] = messageEmbedding == null
                    ? -1f
                    : BertTinyEncoder.cosineSimilarity(queryEmbedding, messageEmbedding);
        }
        Integer[] order = new Integer[olderCount];
        for (int i = 0; i < olderCount; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Float.compare(scores[b], scores[a]));

        Set<Integer> keep = new HashSet<>();
        int selected = 0;
        for (int idx : order) {
            if (selected >= HISTORY_RELEVANCE_MAX_OLDER) break;
            if (scores[idx] < HISTORY_RELEVANCE_MIN_SCORE) break;
            keep.add(idx);
            selected++;
        }
        for (int i = recentStart; i < total; i++) {
            keep.add(i);
        }
        List<Message> filtered = new ArrayList<>();
        for (int i = 0; i < total; i++) {
            if (keep.contains(i)) {
                filtered.add(history.get(i));
            }
        }
        return filtered.isEmpty() ? history : filtered;
    }

    private boolean shouldCacheResponse(String lastUserText, String fullResponse) {
        return semanticResponseCache != null
                && lastUserText != null
                && fullResponse != null
                && !fullResponse.trim().isEmpty();
    }

    /**
     * Runs a full model generation pass for an exact dictionary query.
     *
     * <p>The model is given only the topic (never the definition) together with
     * strict template instructions.  After generation the response is assembled
     * by {@link #assembleDictionaryResponse}: the verbatim definition is injected
     * where the model placed {@value #DICT_DEFINITION_PLACEHOLDER}, and the
     * canonical topic name is enforced in the closing question.
     */
    private void generateDictionaryResponse(
            FactualDictionary.DictionaryMatch dictMatch,
            List<Message> history,
            String summary,
            String archivedSummary,
            Map<String, String> promptTags,
            StreamCallback callback
    ) throws Exception {
        String topic      = dictMatch.topic;
        String definition = dictMatch.definition;

        String prompt     = buildDictionaryPrompt(
                topic, history, summary, archivedSummary, promptTags);
        long[] promptIds  = tokenizer.encodeWithSpecialTokens(prompt);
        promptIds         = trimPromptWithAttentionSink(promptIds);

        GenerationOptions previousOptions = generationOptions;
        generationOptions = new GenerationOptions(
                previousOptions.temperature,
                previousOptions.topP,
                Math.min(previousOptions.maxNewTokens, DICTIONARY_MAX_NEW_TOKENS));

        StringBuilder responseBuilder = new StringBuilder();
        // The generation methods (generateWithKVCache / generateNoKVCache) always
        // accumulate tokens into responseBuilder via out.append(piece) and then call
        // onComplete(out.toString()).  onToken is suppressed here so that the raw
        // (placeholder-containing) tokens are never forwarded to the UI; the fully
        // assembled response is delivered in a single onComplete call instead.
        StreamCallback dictCallback = new StreamCallback() {
            @Override public void onToken(String tokenText) { /* suppress – see comment above */ }

            @Override
            public void onComplete(String fullResponse) {
                callback.onComplete(
                        assembleDictionaryResponse(
                                fullResponse,
                                definition,
                                topic,
                                stopRequested.get()));
            }

            @Override
            public void onError(Exception e) { callback.onError(e); }
        };

        try {
            if (hasKVCache) {
                generateWithKVCache(promptIds, responseBuilder, dictCallback);
            } else {
                generateNoKVCache(promptIds, responseBuilder, dictCallback);
            }
        } finally {
            generationOptions = previousOptions;
        }
    }

    /**
     * Builds a ChatML prompt for a dictionary definition query.
     *
     * <p>The model learns only the {@code topic}; the definition is intentionally
     * withheld.  The system block instructs the model to:
     * <ol>
     *   <li>Write a brief introductory phrase (the <em>beginning</em>).</li>
     *   <li>Output the literal placeholder {@value #DICT_DEFINITION_PLACEHOLDER}
     *       where the definition will be injected.</li>
     *   <li>Ask a closing question that names the exact topic (the <em>ending</em>).</li>
     * </ol>
     */
    private String buildDictionaryPrompt(
            String topic,
            List<Message> history,
            String summary,
            String archivedSummary,
            Map<String, String> promptTags
    ) {
        Map<String, String> effectiveTags = buildRealtimeTags(promptTags);
        StringBuilder sb = new StringBuilder();
        sb.append("<|im_start|>system\n")
          .append("You are a helpful on-device AI assistant.\n")
          .append("The user asked for the definition of: ").append(topic).append(".\n")
          .append("Respond using EXACTLY this three-part format – no exceptions:\n")
          .append("  1. A brief introductory phrase leading into the definition.\n")
          .append("  2. The literal text ").append(DICT_DEFINITION_PLACEHOLDER)
          .append(" (this will be replaced with the actual definition).\n")
          .append("  3. A closing question asking whether the user wants to know more about ")
          .append(topic).append(".\n")
          .append("Rules you MUST follow:\n")
          .append("  - ").append(DICT_DEFINITION_PLACEHOLDER)
          .append(" MUST appear exactly once, unchanged.\n")
          .append("  - The closing question MUST contain the exact topic name: ")
          .append(topic).append(".\n")
          .append("  - You may rephrase parts 1 and 3, but MUST NOT alter ")
          .append(DICT_DEFINITION_PLACEHOLDER).append(" or the topic name.\n")
          .append("  - Do NOT include the actual definition – only the placeholder.\n")
          .append("Routing mode: $route.\n")
          .append("Current date is $date, current time is $time.\n")
          .append("<|im_end|>\n");
        appendSummaryBlock(sb, "Archived conversation summary", archivedSummary);
        appendSummaryBlock(sb, "Conversation summary", summary);
        for (Message msg : history) {
            sb.append(msg.isUser() ? "<|im_start|>user\n" : "<|im_start|>assistant\n")
              .append(msg.getContent()).append('\n')
              .append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return applyPromptTags(sb.toString(), effectiveTags);
    }

    /**
     * Assembles the final dictionary response by injecting the verbatim
     * {@code definition} in place of {@value #DICT_DEFINITION_PLACEHOLDER} and
     * verifying that the closing question contains the exact {@code topic}.
     *
     * <p>If the model did not emit the placeholder the {@code modelOutput} is used
     * as the beginning and the canonical ending is appended:
     * <pre>
     *   [model beginning]
     *
     *   [verbatim definition]
     *
     *   Do you want to know more about [topic]?
     * </pre>
     */
    private static String assembleDictionaryResponse(
            String modelOutput, String definition, String topic, boolean wasStopped) {
        String trimmed = (modelOutput == null) ? "" : modelOutput.trim();
        String ending = DICTIONARY_CLOSING_PHRASE + topic + "?";

        int idx = trimmed.indexOf(DICT_DEFINITION_PLACEHOLDER);
        if (idx >= 0) {
            String intro = sanitizeDictionaryIntro(trimmed.substring(0, idx));
            if (intro.isEmpty()) {
                intro = "Here is the definition of " + topic + ".";
            }
            return (intro + "\n\n" + definition + "\n\n" + ending).trim();
        }

        // Fallback: model omitted placeholder. Keep only an intro-like paraphrase.
        String intro = sanitizeDictionaryIntro(trimmed);
        if (intro.isEmpty()) {
            intro = "Here is the definition of " + topic + ".";
        }
        if (wasStopped && !endsWithSentenceBoundary(intro)) {
            return intro;
        }
        return (intro + "\n\n" + definition + "\n\n" + ending).trim();
    }

    private static String sanitizeDictionaryIntro(String text) {
        if (text == null) return "";
        String normalized = WHITESPACE_PATTERN
                .matcher(text.replace(DICT_DEFINITION_PLACEHOLDER, " "))
                .replaceAll(" ")
                .trim();
        if (normalized.isEmpty()) return "";
        String lower = normalized.toLowerCase(java.util.Locale.ROOT);
        int closingIdx = lower.indexOf(DICTIONARY_CLOSING_PHRASE.toLowerCase(java.util.Locale.ROOT));
        if (closingIdx >= 0) {
            normalized = normalized.substring(0, closingIdx).trim();
        }
        if (normalized.isEmpty()) return "";

        int boundary = findSentenceBoundary(normalized);
        if (boundary > 0) {
            return normalized.substring(0, boundary).trim();
        }
        if (normalized.length() <= MAX_INTRO_CHARS) {
            return normalized;
        }
        int cut = normalized.lastIndexOf(' ', MAX_INTRO_CHARS);
        if (cut <= 0) cut = MAX_INTRO_CHARS;
        return normalized.substring(0, cut).trim();
    }

    private static int findSentenceBoundary(String text) {
        int maxLen = Math.min(text.length(), MAX_INTRO_CHARS);
        for (int i = 0; i < maxLen; i++) {
            char c = text.charAt(i);
            if (c == '.' || c == '!' || c == '?' || c == '\n') {
                // Return the exclusive end index (safe for substring(0, boundary)).
                return i + 1;
            }
        }
        return -1;
    }

    private static boolean endsWithSentenceBoundary(String text) {
        if (text == null) return false;
        String trimmed = text.trim();
        if (trimmed.isEmpty()) return false;
        char last = trimmed.charAt(trimmed.length() - 1);
        return last == '.' || last == '!' || last == '?' || last == '\n';
    }

    private static boolean isMissingPositionIdsError(Exception e) {
        boolean sawOrtException = false;
        Throwable t = e;
        while (t != null) {
            if (t instanceof OrtException) {
                sawOrtException = true;
            }
            String message = t.getMessage();
            if (sawOrtException && message != null
                    && MISSING_POSITION_IDS_PATTERN.matcher(message).find()) {
                return true;
            }
            t = t.getCause();
        }
        return false;
    }

    private String buildDictionaryFacts(List<Message> history) {
        if (factualDictionary == null || history == null || history.isEmpty()) {
            return "";
        }
        Message lastUser = null;
        for (int i = history.size() - 1; i >= 0; i--) {
            Message msg = history.get(i);
            if (msg.isUser()) {
                lastUser = msg;
                break;
            }
        }
        if (lastUser == null) {
            return "";
        }
        List<String> facts = bertEncoder != null
                ? factualDictionary.findSemanticFacts(
                        lastUser.getContent(), MAX_DICTIONARY_FACTS, bertEncoder)
                : factualDictionary.findRelevantFacts(
                        lastUser.getContent(), MAX_DICTIONARY_FACTS);
        if (facts.isEmpty()) {
            return "";
        }
        return String.join(" | ", facts);
    }

    private static Map<String, String> buildRealtimeTags(Map<String, String> promptTags) {
        Map<String, String> merged = new LinkedHashMap<>();
        LocalDateTime now = LocalDateTime.now();
        merged.put(TAG_DATE, DATE_FORMAT.format(now));
        merged.put(TAG_TIME, TIME_FORMAT.format(now));
        merged.put(TAG_DATETIME, DATETIME_FORMAT.format(now));
        merged.put("$route", DEFAULT_ROUTE);
        if (promptTags != null) {
            for (Map.Entry<String, String> entry : promptTags.entrySet()) {
                if (entry.getKey() == null || entry.getValue() == null) continue;
                merged.put(entry.getKey(), entry.getValue());
            }
        }
        return merged;
    }

    private static String applyPromptTags(String prompt, Map<String, String> tags) {
        String resolved = prompt;
        for (Map.Entry<String, String> entry : tags.entrySet()) {
            resolved = resolved.replace(entry.getKey(), entry.getValue());
        }
        return resolved;
    }

    private static void appendSummaryBlock(StringBuilder sb, String label, String summary) {
        if (summary == null) return;
        String trimmed = summary.trim();
        if (trimmed.isEmpty()) return;
        sb.append("<|im_start|>system\n")
          .append(label)
          .append(": ")
          .append(trimmed)
          .append('\n')
          .append("<|im_end|>\n");
    }

    private static long[] trimPromptWithAttentionSink(long[] promptIds) {
        if (promptIds.length <= MAX_CONTEXT_LEN) {
            return promptIds;
        }

        // Keep sink bounded so we always preserve substantial recent context.
        int sinkLen = Math.min(
                ATTENTION_SINK_TOKEN_COUNT,
                MAX_CONTEXT_LEN / ATTENTION_SINK_MAX_CONTEXT_DIVISOR
        );
        int tailLen = MAX_CONTEXT_LEN - sinkLen;
        long[] trimmed = new long[MAX_CONTEXT_LEN];
        System.arraycopy(promptIds, 0, trimmed, 0, sinkLen);
        System.arraycopy(promptIds, promptIds.length - tailLen, trimmed, sinkLen, tailLen);
        return trimmed;
    }

    // -----------------------------------------------------------------
    // Generation with KV-cache (preferred)
    // -----------------------------------------------------------------

    private void generateWithKVCache(long[] promptIds,
                                      StringBuilder out,
                                      StreamCallback cb) throws OrtException {
        int numLayers = pastKeyNames.size();
        // Past KV state: [numLayers][batch=1][numKvHeads][pastSeq][headDim]
        // Represented as flat float arrays with an associated past-length counter.
        float[][] cachedKeys = new float[numLayers][0];
        float[][] cachedVals = new float[numLayers][0];
        int pastLen = 0;

        // Prefill with the whole prompt
        long[] currentIds = promptIds;
        List<Long> generatedIds = new ArrayList<>();
        String streamedText = "";

        GenerationOptions options = generationOptions;
        for (int step = 0; step < options.maxNewTokens; step++) {
            if (stopRequested.get()) break;
            int seqLen   = currentIds.length;
            int totalLen = pastLen + seqLen;
            long[] attnMask = new long[totalLen];
            Arrays.fill(attnMask, 1L);

            StepResult sr = runKVStep(currentIds, attnMask, pastLen,
                                       cachedKeys, cachedVals, numLayers);

            // Update cache
            cachedKeys = sr.newKeys;
            cachedVals = sr.newVals;
            pastLen    = totalLen;

            long nextToken = selectNextToken(
                    sr.lastLogits,
                    buildContextIds(promptIds, generatedIds),
                    generatedIds,
                    options.temperature,
                    options.topP);
            if (nextToken == BPETokenizer.TOKEN_IM_END ||
                    nextToken == BPETokenizer.TOKEN_EOS) break;

            generatedIds.add(nextToken);
            String decodedSoFar = tokenizer.decode(generatedIds);
            // Rare tokenizer boundary fallback: if the incremental prefix check fails,
            // stream the full decoded segment once to avoid dropping generated text.
            String piece = decodedSoFar.startsWith(streamedText)
                    ? decodedSoFar.substring(streamedText.length())
                    : decodedSoFar;
            if (!piece.isEmpty()) {
                out.append(piece);
                cb.onToken(piece);
                streamedText = decodedSoFar;
            }

            // Next step: only feed the generated token
            currentIds = new long[]{nextToken};
        }

        cb.onComplete(out.toString());
    }

    private StepResult runKVStep(long[] inputIds, long[] attnMask,
                                  int pastLen,
                                  float[][] cachedKeys, float[][] cachedVals,
                                  int numLayers) throws OrtException {

        Map<String, OnnxTensor> inputs = new HashMap<>();
        long[] idShape   = {1, inputIds.length};
        long[] maskShape = {1, attnMask.length};
        long[] kvShape   = {1, NUM_KV_HEADS, pastLen, HEAD_DIM};

        inputs.put("input_ids",
                   OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds),  idShape));
        inputs.put("attention_mask",
                    OnnxTensor.createTensor(env, LongBuffer.wrap(attnMask), maskShape));
        if (requiresPositionIds) {
            long[] pos = new long[inputIds.length];
            for (int i = 0; i < inputIds.length; i++) pos[i] = pastLen + i;
            inputs.put("position_ids",
                    OnnxTensor.createTensor(env, LongBuffer.wrap(pos), idShape));
        }

        for (int i = 0; i < numLayers; i++) {
            inputs.put(pastKeyNames.get(i),
                       OnnxTensor.createTensor(env,
                               FloatBuffer.wrap(cachedKeys[i]), kvShape));
            inputs.put(pastValNames.get(i),
                       OnnxTensor.createTensor(env,
                               FloatBuffer.wrap(cachedVals[i]), kvShape));
        }

        StepResult result = new StepResult(numLayers);
        try (OrtSession.Result ort = session.run(inputs)) {
            float[][][] logits = (float[][][]) ort.get("logits").get().getValue();
            result.lastLogits = logits[0][inputIds.length - 1];

            int newTotal = pastLen + inputIds.length;
            for (int i = 0; i < numLayers; i++) {
                if (ort.get(presKeyNames.get(i)).isPresent()) {
                    result.newKeys[i] = flattenKVTensor(
                            (float[][][][]) ort.get(presKeyNames.get(i)).get().getValue(),
                            newTotal);
                    result.newVals[i] = flattenKVTensor(
                            (float[][][][]) ort.get(presValNames.get(i)).get().getValue(),
                            newTotal);
                }
            }
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
        return result;
    }

    // -----------------------------------------------------------------
    // Generation without KV-cache (fallback for simpler ONNX exports)
    // -----------------------------------------------------------------

    private void generateNoKVCache(long[] promptIds,
                                    StringBuilder out,
                                    StreamCallback cb) throws OrtException {
        List<Long> allIds = new ArrayList<>();
        for (long id : promptIds) allIds.add(id);
        List<Long> generatedIds = new ArrayList<>();
        String streamedText = "";

        GenerationOptions options = generationOptions;
        for (int step = 0; step < options.maxNewTokens; step++) {
            if (stopRequested.get()) break;
            long[] ids  = toLongArray(allIds);
            long[] mask = new long[ids.length];
            Arrays.fill(mask, 1L);

            long[] idShape   = {1, ids.length};
            long[] maskShape = {1, ids.length};

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids",
                       OnnxTensor.createTensor(env, LongBuffer.wrap(ids),  idShape));
            inputs.put("attention_mask",
                       OnnxTensor.createTensor(env, LongBuffer.wrap(mask), maskShape));
            if (requiresPositionIds) {
                long[] pos = new long[ids.length];
                for (int i = 0; i < ids.length; i++) pos[i] = i;
                inputs.put("position_ids",
                        OnnxTensor.createTensor(env, LongBuffer.wrap(pos), idShape));
            }

            long nextToken;
            try (OrtSession.Result ort = session.run(inputs)) {
                float[][][] logits = (float[][][]) ort.get("logits").get().getValue();
                nextToken = selectNextToken(
                        logits[0][ids.length - 1],
                        ids,
                        generatedIds,
                        options.temperature,
                        options.topP);
            } finally {
                for (OnnxTensor t : inputs.values()) t.close();
            }

            if (nextToken == BPETokenizer.TOKEN_IM_END ||
                    nextToken == BPETokenizer.TOKEN_EOS) break;

            allIds.add(nextToken);
            generatedIds.add(nextToken);
            String decodedSoFar = tokenizer.decode(generatedIds);
            // Rare tokenizer boundary fallback: if the incremental prefix check fails,
            // stream the full decoded segment once to avoid dropping generated text.
            String piece = decodedSoFar.startsWith(streamedText)
                    ? decodedSoFar.substring(streamedText.length())
                    : decodedSoFar;
            if (!piece.isEmpty()) {
                out.append(piece);
                cb.onToken(piece);
                streamedText = decodedSoFar;
            }
        }

        cb.onComplete(out.toString());
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    /** Flatten a [1, numKvHeads, seqLen, headDim] tensor to a 1-D float array. */
    private static float[] flattenKVTensor(float[][][][] kv, int seqLen) {
        if (kv == null || kv.length == 0) return new float[0];
        int size = NUM_KV_HEADS * seqLen * HEAD_DIM;
        float[] flat = new float[size];
        int idx = 0;
        // kv shape: [batch=1, numKvHeads, seqLen, headDim]
        // kv[0] is float[][][] (numKvHeads × seqLen × headDim)
        for (float[][] headArr : kv[0]) {  // headArr: [seqLen][headDim]
            for (float[] seqArr : headArr) {  // seqArr: [headDim]
                for (float val : seqArr) {
                    if (idx < flat.length) flat[idx++] = val;
                }
            }
        }
        return flat;
    }

    private static long argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    private static long selectNextToken(
            float[] logits,
            long[] contextIds,
            List<Long> generatedIds,
            float temperature,
            float topP
    ) {
        float[] adjusted = Arrays.copyOf(logits, logits.length);
        applyRepetitionPenalty(adjusted, contextIds);
        applyNoRepeatNgram(adjusted, generatedIds);
        if (temperature <= MIN_TEMPERATURE + FLOAT_EPSILON
                || topP <= MIN_TOP_P + FLOAT_EPSILON) {
            return argmax(adjusted);
        }
        return sampleTopP(adjusted, temperature, topP);
    }

    private static long sampleTopP(float[] logits, float temperature, float topP) {
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (float logit : logits) {
            if (logit > maxLogit) maxLogit = logit;
        }

        double[] probabilities = new double[logits.length];
        double sum = 0d;
        for (int i = 0; i < logits.length; i++) {
            float logit = logits[i];
            if (!Float.isFinite(logit)) {
                probabilities[i] = 0d;
                continue;
            }
            double scaled = (logit - maxLogit) / temperature;
            double value = Math.exp(scaled);
            probabilities[i] = value;
            sum += value;
        }
        if (!(sum > 0d)) {
            return argmax(logits);
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }

        List<Integer> sorted = new ArrayList<>(probabilities.length);
        for (int i = 0; i < probabilities.length; i++) {
            sorted.add(i);
        }
        sorted.sort((a, b) -> Double.compare(probabilities[b], probabilities[a]));

        double cumulative = 0d;
        int cutoff = sorted.size();
        for (int i = 0; i < sorted.size(); i++) {
            cumulative += probabilities[sorted.get(i)];
            if (cumulative >= topP) {
                cutoff = i + 1;
                break;
            }
        }

        if (cutoff <= 0) {
            return sorted.get(0);
        }
        double sample = ThreadLocalRandom.current().nextDouble();
        double running = 0d;
        double topPSum = 0d;
        for (int i = 0; i < cutoff; i++) {
            topPSum += probabilities[sorted.get(i)];
        }
        if (!(topPSum > 0d)) {
            return sorted.get(0);
        }

        for (int i = 0; i < cutoff; i++) {
            int token = sorted.get(i);
            running += probabilities[token] / topPSum;
            if (sample <= running) {
                return token;
            }
        }
        return sorted.get(cutoff - 1);
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void applyRepetitionPenalty(float[] logits, long[] contextIds) {
        Set<Long> recentTokenIds = new HashSet<>();

        int contextStart = Math.max(0, contextIds.length - REPETITION_WINDOW);
        for (int i = contextStart; i < contextIds.length; i++) {
            recentTokenIds.add(contextIds[i]);
        }

        for (Long tokenId : recentTokenIds) {
            int index = tokenId.intValue();
            if (index < 0 || index >= logits.length) continue;
            // Same logit-space repetition penalty used by common Transformer generation:
            // positive logits are divided, while negative logits are multiplied, so repeated
            // tokens always become less likely regardless of sign.
            logits[index] = logits[index] > 0f
                    ? logits[index] / REPETITION_PENALTY
                    : logits[index] * REPETITION_PENALTY;
        }
    }

    private static void applyNoRepeatNgram(float[] logits, List<Long> generatedIds) {
        if (generatedIds.size() < NO_REPEAT_NGRAM_SIZE - 1) return;
        int prefixLen = NO_REPEAT_NGRAM_SIZE - 1;
        Map<String, Set<Long>> seenContinuations = new HashMap<>();
        for (int i = 0; i <= generatedIds.size() - NO_REPEAT_NGRAM_SIZE; i++) {
            String key = buildNgramPrefixKey(generatedIds, i, prefixLen);
            long continuation = generatedIds.get(i + prefixLen);
            seenContinuations.computeIfAbsent(key, ignored -> new HashSet<>()).add(continuation);
        }

        String currentPrefix = buildNgramPrefixKey(
                generatedIds, generatedIds.size() - prefixLen, prefixLen);
        Set<Long> bannedContinuations = seenContinuations.get(currentPrefix);
        if (bannedContinuations == null) return;
        for (Long bannedToken : bannedContinuations) {
            int banned = bannedToken.intValue();
            if (banned >= 0 && banned < logits.length) {
                logits[banned] = Float.NEGATIVE_INFINITY;
            }
        }
    }

    private static String buildNgramPrefixKey(List<Long> tokens, int start, int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            if (i > 0) sb.append(',');
            sb.append(tokens.get(start + i));
        }
        return sb.toString();
    }

    private static long[] buildContextIds(long[] promptIds, List<Long> generatedIds) {
        long[] context = Arrays.copyOf(promptIds, promptIds.length + generatedIds.size());
        for (int i = 0; i < generatedIds.size(); i++) {
            context[promptIds.length + i] = generatedIds.get(i);
        }
        return context;
    }

    private static long[] toLongArray(List<Long> list) {
        long[] arr = new long[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return arr;
    }

    public void close() {
        try { session.close(); } catch (OrtException e) { Log.e(TAG, "close session", e); }
        env.close();
    }

    public void requestStop() {
        stopRequested.set(true);
    }

    // -----------------------------------------------------------------
    // Inner classes
    // -----------------------------------------------------------------

    private static class StepResult {
        float[] lastLogits;
        final float[][] newKeys;
        final float[][] newVals;

        StepResult(int numLayers) {
            newKeys = new float[numLayers][0];
            newVals = new float[numLayers][0];
        }
    }
}
