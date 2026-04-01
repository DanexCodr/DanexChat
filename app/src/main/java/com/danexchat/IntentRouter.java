package com.danexchat;

import android.util.Log;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Semantic query intent router powered by BERT-tiny embeddings.
 *
 * Classifies each user query into one of six intent categories using a
 * nearest-prototype approach over BERT-tiny sentence embeddings:
 * <ul>
 *   <li><b>factual</b> – knowledge lookups, definitions, history</li>
 *   <li><b>creative</b> – writing, storytelling, imagination</li>
 *   <li><b>coding</b> – programming, algorithms, debugging</li>
 *   <li><b>math</b> – arithmetic, equations, calculations</li>
 *   <li><b>conversational</b> – greetings, small talk</li>
 *   <li><b>general</b> – catch-all for unclear intent</li>
 * </ul>
 *
 * Falls back to keyword-based routing when the BERT-tiny encoder is
 * unavailable or when the best semantic match is below the confidence
 * threshold. Each category maps to a distinct
 * {@link SmolLMInference.GenerationOptions} profile (temperature, top-p,
 * max tokens) tuned for that intent.
 *
 * {@link #route} is designed to be called on a background thread.
 */
public class IntentRouter {

    private static final String TAG = "IntentRouter";

    public static final String ROUTE_FACTUAL        = "factual";
    public static final String ROUTE_CREATIVE       = "creative";
    public static final String ROUTE_CODING         = "coding";
    public static final String ROUTE_MATH           = "math";
    public static final String ROUTE_CONVERSATIONAL = "conversational";
    public static final String ROUTE_GENERAL        = "general";

    // Minimum cosine similarity to trust the BERT classifier;
    // queries below this threshold fall back to keyword matching.
    private static final float BERT_CONFIDENCE_THRESHOLD = 0.25f;

    // Per-route generation parameters
    private static final float TEMP_ZERO     = 0.0f;
    private static final float TEMP_CREATIVE = 0.7f;
    private static final float TEMP_CASUAL   = 0.5f;

    // Keyword fallback patterns (richer than the previous two-pattern set)
    private static final Pattern FACTUAL_PATTERN = Pattern.compile(
            "(?i)\\b(?:who|what|when|where|which|how many|define|explain|history|"
            + "fact|today|day|year|century|born|died|located|capital|population)\\b");
    private static final Pattern CREATIVE_PATTERN = Pattern.compile(
            "(?i)\\b(?:write|create|story|poem|imagine|brainstorm|roleplay|"
            + "fiction|novel|song|lyric|narrative|compose|draft)\\b");
    private static final Pattern CODING_PATTERN = Pattern.compile(
            "(?i)\\b(?:code|function|class|implement|debug|program|algorithm|"
            + "script|loop|variable|java|python|kotlin|javascript|sql|bug|error)\\b");
    private static final Pattern MATH_PATTERN = Pattern.compile(
            "(?i)\\b(?:calculate|compute|solve|equation|sum|plus|minus|multiply|"
            + "divide|integral|derivative|percent|ratio|probability|formula)\\b");
    private static final Pattern CONVERSATIONAL_PATTERN = Pattern.compile(
            "(?i)\\b(?:hello|hi|hey|thanks|thank you|how are you|"
            + "good morning|good night|bye|goodbye|chat|help me)\\b");

    // Prototype phrase sets used to build mean embeddings per category
    private static final String[] ROUTE_NAMES = {
        ROUTE_FACTUAL, ROUTE_CREATIVE, ROUTE_CODING, ROUTE_MATH, ROUTE_CONVERSATIONAL
    };

    private static final String[][] ROUTE_PROTOTYPES = {
        // factual
        {
            "what is the definition of this concept",
            "who invented the telephone and when",
            "explain how photosynthesis works",
            "what are the historical facts about rome",
            "when did the second world war end"
        },
        // creative
        {
            "write a short story about a lost robot",
            "compose a poem about the ocean",
            "imagine a world without gravity",
            "roleplay as a medieval knight",
            "brainstorm ideas for a science fiction novel"
        },
        // coding
        {
            "write a java function to reverse a string",
            "implement a binary search algorithm",
            "debug this python code that crashes",
            "how do I create a loop in kotlin",
            "write a sql query to find duplicates"
        },
        // math
        {
            "calculate the square root of 144",
            "solve this quadratic equation for x",
            "what is twenty percent of three hundred",
            "compute the derivative of x squared",
            "find the probability of rolling a six"
        },
        // conversational
        {
            "hello how are you doing today",
            "thank you for your help",
            "good morning have a nice day",
            "can you chat with me for a while",
            "tell me something fun and interesting"
        }
    };

    private final BertTinyEncoder encoder;
    private final Map<String, float[]> prototypeEmbeddings = new HashMap<>();

    public IntentRouter(BertTinyEncoder encoder) {
        this.encoder = encoder;
        if (encoder != null) {
            buildPrototypeEmbeddings();
        }
    }

    /** Immutable result of intent routing. */
    public static final class RouteResult {
        public final String route;
        public final SmolLMInference.GenerationOptions options;

        RouteResult(String route, SmolLMInference.GenerationOptions options) {
            this.route   = route;
            this.options = options;
        }
    }

    /**
     * Classify {@code query} and return the detected intent route together
     * with the optimal {@link SmolLMInference.GenerationOptions} for that route.
     * Designed to be called on a background thread.
     */
    public RouteResult route(String query) {
        String routeName = classify(query);
        return new RouteResult(routeName, optionsFor(routeName));
    }

    // -----------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------

    private String classify(String query) {
        if (encoder != null && !prototypeEmbeddings.isEmpty()) {
            return classifyWithBert(query);
        }
        return classifyWithKeywords(query);
    }

    private String classifyWithBert(String query) {
        float[] queryEmb = encoder.encode(query);
        if (queryEmb == null) {
            return classifyWithKeywords(query);
        }
        String bestRoute = ROUTE_GENERAL;
        float bestScore  = -2f;
        for (String name : ROUTE_NAMES) {
            float[] proto = prototypeEmbeddings.get(name);
            if (proto == null) continue;
            float score = BertTinyEncoder.cosineSimilarity(queryEmb, proto);
            if (score > bestScore) {
                bestScore = score;
                bestRoute = name;
            }
        }
        // If confidence is low, keyword rules may give a stronger signal.
        if (bestScore < BERT_CONFIDENCE_THRESHOLD) {
            String kwRoute = classifyWithKeywords(query);
            return ROUTE_GENERAL.equals(kwRoute) ? bestRoute : kwRoute;
        }
        return bestRoute;
    }

    private static String classifyWithKeywords(String query) {
        if (FACTUAL_PATTERN.matcher(query).find())        return ROUTE_FACTUAL;
        if (CODING_PATTERN.matcher(query).find())         return ROUTE_CODING;
        if (MATH_PATTERN.matcher(query).find())           return ROUTE_MATH;
        if (CREATIVE_PATTERN.matcher(query).find())       return ROUTE_CREATIVE;
        if (CONVERSATIONAL_PATTERN.matcher(query).find()) return ROUTE_CONVERSATIONAL;
        return ROUTE_GENERAL;
    }

    private static SmolLMInference.GenerationOptions optionsFor(String route) {
        switch (route) {
            case ROUTE_FACTUAL:
                return new SmolLMInference.GenerationOptions(TEMP_ZERO,     0.82f, 220);
            case ROUTE_CREATIVE:
                return new SmolLMInference.GenerationOptions(TEMP_CREATIVE,  0.95f, 400);
            case ROUTE_CODING:
                return new SmolLMInference.GenerationOptions(TEMP_ZERO,     0.85f, 350);
            case ROUTE_MATH:
                return new SmolLMInference.GenerationOptions(TEMP_ZERO,     0.80f, 180);
            case ROUTE_CONVERSATIONAL:
                return new SmolLMInference.GenerationOptions(TEMP_CASUAL,   0.90f, 160);
            default:
                return SmolLMInference.GenerationOptions.defaults();
        }
    }

    /**
     * Pre-compute a mean L2-normalized prototype embedding for each route.
     * Called once at construction time when the encoder is available.
     */
    private void buildPrototypeEmbeddings() {
        for (int r = 0; r < ROUTE_NAMES.length; r++) {
            String name    = ROUTE_NAMES[r];
            String[] phrases = ROUTE_PROTOTYPES[r];
            float[] mean   = null;
            int count      = 0;
            for (String phrase : phrases) {
                float[] emb = encoder.encode(phrase);
                if (emb == null) continue;
                if (mean == null) mean = new float[emb.length];
                for (int i = 0; i < mean.length && i < emb.length; i++) {
                    mean[i] += emb[i];
                }
                count++;
            }
            if (mean == null || count == 0) continue;
            for (int i = 0; i < mean.length; i++) mean[i] /= count;
            // Re-normalize the mean to get a proper unit vector
            float sumSq = 0f;
            for (float v : mean) sumSq += v * v;
            if (sumSq > 0f) {
                float norm = (float) Math.sqrt(sumSq);
                for (int i = 0; i < mean.length; i++) mean[i] /= norm;
            }
            prototypeEmbeddings.put(name, mean);
            Log.d(TAG, "Prototype embedding ready for route: " + name);
        }
    }
}
