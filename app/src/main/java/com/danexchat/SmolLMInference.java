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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
    private static final int MAX_CONTEXT_LEN = 2048;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final BPETokenizer tokenizer;

    // KV-cache tensor name lists (populated at load time)
    private final List<String> pastKeyNames = new ArrayList<>();
    private final List<String> pastValNames = new ArrayList<>();
    private final List<String> presKeyNames = new ArrayList<>();
    private final List<String> presValNames = new ArrayList<>();
    private boolean hasKVCache = false;
    private boolean requiresPositionIds = false;

    // -----------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------

    public SmolLMInference(File modelFile, File tokenizerFile)
            throws OrtException, IOException, org.json.JSONException {

        env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(
                Math.min(4, Runtime.getRuntime().availableProcessors()));

        session = env.createSession(modelFile.getAbsolutePath(), opts);
        tokenizer = new BPETokenizer(tokenizerFile);
        discoverKVNames();
        Log.i(TAG, "Loaded SmolLM2. KV-cache layers: " + pastKeyNames.size());
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

    /** Build the SmolLM2-Instruct chat prompt (ChatML format). */
    public String buildPrompt(List<Message> history) {
        StringBuilder sb = new StringBuilder();
        sb.append("<|im_start|>system\n")
          .append("You are SmolLM, a helpful AI assistant trained by Hugging Face.\n")
          .append("<|im_end|>\n");
        for (Message msg : history) {
            sb.append(msg.isUser() ? "<|im_start|>user\n" : "<|im_start|>assistant\n")
              .append(msg.getContent()).append('\n')
              .append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }

    /**
     * Generate a response for the given conversation history.
     * Must be called on a background thread.
     */
    public synchronized void generate(List<Message> history, StreamCallback callback) {
        try {
            generateInternal(history, callback);
        } catch (Exception firstError) {
            if (!requiresPositionIds && isMissingPositionIdsError(firstError)) {
                Log.w(TAG, "Retrying generation with position_ids enabled after runtime error", firstError);
                requiresPositionIds = true;
                try {
                    generateInternal(history, callback);
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

    private void generateInternal(List<Message> history, StreamCallback callback) throws Exception {
        String prompt   = buildPrompt(history);
        long[] promptIds = tokenizer.encodeWithSpecialTokens(prompt);

        // Trim to max context
        if (promptIds.length > MAX_CONTEXT_LEN) {
            promptIds = Arrays.copyOfRange(
                    promptIds, promptIds.length - MAX_CONTEXT_LEN, promptIds.length);
        }

        StringBuilder responseBuilder = new StringBuilder();
        if (hasKVCache) {
            generateWithKVCache(promptIds, responseBuilder, callback);
        } else {
            generateNoKVCache(promptIds, responseBuilder, callback);
        }
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

        for (int step = 0; step < MAX_NEW_TOKENS; step++) {
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

            long nextToken = sr.nextToken;
            if (nextToken == BPETokenizer.TOKEN_IM_END ||
                    nextToken == BPETokenizer.TOKEN_EOS) break;

            generatedIds.add(nextToken);
            String decodedSoFar = tokenizer.decode(generatedIds);
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
            result.nextToken = argmax(logits[0][inputIds.length - 1]);

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

        for (int step = 0; step < MAX_NEW_TOKENS; step++) {
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
                nextToken = argmax(logits[0][ids.length - 1]);
            } finally {
                for (OnnxTensor t : inputs.values()) t.close();
            }

            if (nextToken == BPETokenizer.TOKEN_IM_END ||
                    nextToken == BPETokenizer.TOKEN_EOS) break;

            allIds.add(nextToken);
            generatedIds.add(nextToken);
            String decodedSoFar = tokenizer.decode(generatedIds);
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

    private static long[] toLongArray(List<Long> list) {
        long[] arr = new long[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return arr;
    }

    public void close() {
        try { session.close(); } catch (OrtException e) { Log.e(TAG, "close session", e); }
        env.close();
    }

    // -----------------------------------------------------------------
    // Inner classes
    // -----------------------------------------------------------------

    private static class StepResult {
        long nextToken;
        final float[][] newKeys;
        final float[][] newVals;

        StepResult(int numLayers) {
            newKeys = new float[numLayers][0];
            newVals = new float[numLayers][0];
        }
    }
}
