package com.danexchat;

import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.File;
import java.io.IOException;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * On-device BERT-tiny encoder using ONNX Runtime.
 *
 * Produces L2-normalized 128-dimensional sentence embeddings from the CLS
 * token of the final hidden state. These embeddings power:
 * <ul>
 *   <li>Semantic intent routing via {@link IntentRouter}</li>
 *   <li>Semantic fact retrieval via {@link FactualDictionary#findSemanticFacts}</li>
 * </ul>
 *
 * All public methods return {@code null}/zero gracefully on any error, so
 * callers should fall back to keyword-based alternatives when {@code null}
 * is returned.
 *
 * Must be called from a background thread for inference.
 */
public class BertTinyEncoder {

    private static final String TAG = "BertTinyEncoder";

    /** BERT-tiny hidden size (output embedding dimension). */
    static final int HIDDEN_SIZE = 128;
    private static final int MAX_SEQ_LEN = 64;
    // Keep BERT lightweight to avoid CPU contention with SmolLM generation.
    private static final int MAX_INTRA_OP_THREADS = 2;

    private static final String OUTPUT_LAST_HIDDEN = "last_hidden_state";
    private static final String INPUT_IDS          = "input_ids";
    private static final String ATTENTION_MASK     = "attention_mask";
    private static final String TOKEN_TYPE_IDS     = "token_type_ids";

    private final OrtEnvironment env;
    private final OrtSession session;
    private final BertTinyTokenizer tokenizer;
    private final boolean requiresTokenTypeIds;

    public BertTinyEncoder(File modelFile, File vocabFile)
            throws OrtException, IOException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(Math.min(MAX_INTRA_OP_THREADS, Runtime.getRuntime().availableProcessors()));
        session = env.createSession(modelFile.getAbsolutePath(), opts);
        tokenizer = new BertTinyTokenizer(vocabFile);
        requiresTokenTypeIds = session.getInputNames().contains(TOKEN_TYPE_IDS);
        Log.i(TAG, "Loaded BERT-tiny. token_type_ids required: " + requiresTokenTypeIds);
    }

    /**
     * Encode {@code text} into a L2-normalized {@value #HIDDEN_SIZE}-dim embedding.
     * Returns {@code null} on any error; callers should fall back to keyword methods.
     */
    public float[] encode(String text) {
        if (text == null || text.trim().isEmpty()) return null;
        try {
            long[] inputIds = tokenizer.encode(text, MAX_SEQ_LEN);
            int seqLen = inputIds.length;
            long[] attnMask = new long[seqLen];
            Arrays.fill(attnMask, 1L);
            long[] shape = {1, seqLen};

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(INPUT_IDS,
                    OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape));
            inputs.put(ATTENTION_MASK,
                    OnnxTensor.createTensor(env, LongBuffer.wrap(attnMask), shape));
            if (requiresTokenTypeIds) {
                long[] typeIds = new long[seqLen]; // all zeros for single-segment input
                inputs.put(TOKEN_TYPE_IDS,
                        OnnxTensor.createTensor(env, LongBuffer.wrap(typeIds), shape));
            }

            float[] embedding;
            try (OrtSession.Result result = session.run(inputs)) {
                // Extract CLS token (position 0) from last_hidden_state[batch=0][cls=0][:]
                float[][][] hidden =
                        (float[][][]) result.get(OUTPUT_LAST_HIDDEN).get().getValue();
                embedding = Arrays.copyOf(hidden[0][0], HIDDEN_SIZE);
            } finally {
                for (OnnxTensor t : inputs.values()) t.close();
            }
            return l2Normalize(embedding);
        } catch (OrtException e) {
            Log.e(TAG, "BERT-tiny encode failed", e);
            return null;
        }
    }

    /**
     * Cosine similarity between two L2-normalized embedding vectors.
     * Returns {@code 0} when either vector is {@code null} or dimensions differ.
     */
    public static float cosineSimilarity(float[] a, float[] b) {
        if (a == null || b == null || a.length != b.length) return 0f;
        float dot = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return Math.max(-1f, Math.min(1f, dot));
    }

    public void close() {
        try { session.close(); } catch (OrtException e) { Log.e(TAG, "close session", e); }
        env.close();
    }

    private static float[] l2Normalize(float[] v) {
        float sumSq = 0f;
        for (float x : v) sumSq += x * x;
        if (sumSq <= 0f) return v;
        float norm = (float) Math.sqrt(sumSq);
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] / norm;
        return out;
    }
}
