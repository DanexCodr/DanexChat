package com.danexchat;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Lightweight semantic cache keyed by BERT-tiny query embeddings.
 * Returns a cached response when a new query is very close in embedding space.
 */
public class SemanticResponseCache {
    private static final int MAX_ENTRIES = 64;
    private static final float HIT_THRESHOLD = 0.93f;

    private final BertTinyEncoder encoder;
    private final Deque<Entry> entries = new ArrayDeque<>();

    public SemanticResponseCache(BertTinyEncoder encoder) {
        this.encoder = encoder;
    }

    public String find(String query) {
        if (encoder == null || query == null || query.trim().isEmpty()) {
            return null;
        }
        float[] queryEmbedding = encoder.encode(query);
        if (queryEmbedding == null) {
            return null;
        }
        String best = null;
        float bestScore = HIT_THRESHOLD;
        for (Entry entry : entries) {
            float score = BertTinyEncoder.cosineSimilarity(queryEmbedding, entry.queryEmbedding);
            if (score >= bestScore) {
                bestScore = score;
                best = entry.response;
            }
        }
        return best;
    }

    public void put(String query, String response) {
        if (encoder == null || query == null || response == null || response.trim().isEmpty()) {
            return;
        }
        float[] queryEmbedding = encoder.encode(query);
        if (queryEmbedding == null) {
            return;
        }
        if (entries.size() >= MAX_ENTRIES) {
            entries.pollFirst();
        }
        entries.addLast(new Entry(queryEmbedding, response));
    }

    public void clear() {
        entries.clear();
    }

    private static final class Entry {
        final float[] queryEmbedding;
        final String response;

        Entry(float[] queryEmbedding, String response) {
            this.queryEmbedding = queryEmbedding;
            this.response = response;
        }
    }
}
