package com.danexchat;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * Lightweight local WordNet-style dictionary used to ground factual prompts.
 *
 * Supports both keyword-based retrieval ({@link #findRelevantFacts}) and
 * BERT-tiny semantic re-ranking ({@link #findSemanticFacts}) when a
 * {@link BertTinyEncoder} is provided. Semantic results are cached across
 * calls to amortize encoding cost over repeated queries.
 */
public class FactualDictionary {

    static final String[] DEFINITION_PREFIXES = {
            "what is ", "what are ", "who is ", "who are ", "define ", "tell me about "
    };
    private static final long MAX_DICTIONARY_BYTES = 1024L * 1024L;
    // Prefer direct key hits, then semantic-ish overlap on key terms, then weaker overlap on definitions.
    private static final int WHOLE_KEY_MATCH_WEIGHT = 4;
    private static final int KEY_TOKEN_OVERLAP_WEIGHT = 3;
    private static final int VALUE_TOKEN_OVERLAP_WEIGHT = 1;
    private static final int MIN_TOKEN_LENGTH = 2;
    private static final Set<String> INVARIANT_IES_WORDS = new HashSet<>(
            java.util.Arrays.asList("species", "series", "aries"));
    private static final Set<String> INVARIANT_AS_WORDS = new HashSet<>(
            java.util.Arrays.asList("atlas", "canvas", "bias", "gas", "yes"));
    private static final Set<String> INVARIANT_OS_WORDS = new HashSet<>(
            java.util.Arrays.asList("chaos", "ethos", "pathos", "cosmos"));
    private static final Pattern LEADING_ARTICLES_PATTERN = Pattern.compile("^(?:(?:a|an|the)\\s+)+");
    // Number of keyword-scored candidates passed to BERT re-ranker
    private static final int SEMANTIC_RERANK_TOP_K = 20;
    // Weight applied to cosine similarity boost on top of keyword score
    private static final float SEMANTIC_BOOST_WEIGHT = 2.0f;

    private final Map<String, String> entries;
    // Lazy embedding cache for dictionary entries; populated during semantic lookups.
    // Access is not synchronized; callers (SmolLMInference.generate) are synchronized.
    private final Map<String, float[]> embeddingCache = new java.util.HashMap<>();

    public FactualDictionary(File dictionaryFile) throws IOException, JSONException {
        if (dictionaryFile.length() > MAX_DICTIONARY_BYTES) {
            throw new IOException("Dictionary asset is too large: " + dictionaryFile.length());
        }
        String json = new String(Files.readAllBytes(dictionaryFile.toPath()), StandardCharsets.UTF_8);
        JSONObject root = new JSONObject(json);
        Map<String, String> loaded = new TreeMap<>();
        Iterator<String> keys = root.keys();
        while (keys.hasNext()) {
            String rawKey = keys.next();
            String key = normalize(rawKey);
            if (key.isEmpty()) continue;
            String value = root.optString(rawKey, "").trim();
            if (value.isEmpty()) continue;
            loaded.put(key, value);
        }
        entries = Collections.unmodifiableMap(loaded);
    }

    public List<String> findRelevantFacts(String text, int maxFacts) {
        if (text == null || text.trim().isEmpty() || maxFacts <= 0 || entries.isEmpty()) {
            return Collections.emptyList();
        }
        String normalizedText = normalize(text);
        if (normalizedText.isEmpty()) {
            return Collections.emptyList();
        }
        Set<String> queryTokens = toTokenSet(normalizedText);
        List<ScoredFact> scoredFacts = new ArrayList<>();
        for (Map.Entry<String, String> entry : entries.entrySet()) {
            String key = entry.getKey();
            int score = 0;
            if (containsWholeWord(normalizedText, key)) {
                score += WHOLE_KEY_MATCH_WEIGHT;
            }
            Set<String> keyTokens = toTokenSet(key);
            score += countOverlap(queryTokens, keyTokens) * KEY_TOKEN_OVERLAP_WEIGHT;
            Set<String> valueTokens = toTokenSet(normalize(entry.getValue()));
            score += countOverlap(queryTokens, valueTokens) * VALUE_TOKEN_OVERLAP_WEIGHT;
            if (score > 0) {
                scoredFacts.add(new ScoredFact(key, entry.getValue(), score));
            }
        }
        scoredFacts.sort(Comparator
                .comparingInt((ScoredFact fact) -> fact.score)
                .reversed()
                .thenComparing(fact -> fact.key));
        List<String> matches = new ArrayList<>(Math.min(maxFacts, scoredFacts.size()));
        for (int i = 0; i < scoredFacts.size() && matches.size() < maxFacts; i++) {
            ScoredFact fact = scoredFacts.get(i);
            matches.add(fact.key + ": " + fact.value);
        }
        return matches;
    }

    /**
     * Retrieve facts semantically relevant to {@code text} by combining
     * keyword scoring with BERT-tiny cosine similarity re-ranking.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>Compute keyword score for every entry (same as {@link #findRelevantFacts}).</li>
     *   <li>Take the top-{@value #SEMANTIC_RERANK_TOP_K} candidates by keyword score.</li>
     *   <li>Re-rank those candidates using BERT cosine similarity between the
     *       query embedding and each entry's cached key+value embedding.</li>
     *   <li>Return the top {@code maxFacts} results by combined score.</li>
     * </ol>
     *
     * <p>Falls back to {@link #findRelevantFacts} when {@code encoder} is {@code null}
     * or when BERT encoding fails.
     *
     * <p>Entry embeddings are cached across calls to amortize the per-call cost.
     */
    public List<String> findSemanticFacts(String text, int maxFacts, BertTinyEncoder encoder) {
        if (encoder == null) {
            return findRelevantFacts(text, maxFacts);
        }
        if (text == null || text.trim().isEmpty() || maxFacts <= 0 || entries.isEmpty()) {
            return Collections.emptyList();
        }
        String normalizedText = normalize(text);
        if (normalizedText.isEmpty()) return Collections.emptyList();

        // Step 1: keyword scoring
        Set<String> queryTokens = toTokenSet(normalizedText);
        List<ScoredFact> candidates = new ArrayList<>();
        for (Map.Entry<String, String> entry : entries.entrySet()) {
            String key = entry.getKey();
            int score  = 0;
            if (containsWholeWord(normalizedText, key)) score += WHOLE_KEY_MATCH_WEIGHT;
            score += countOverlap(queryTokens, toTokenSet(key)) * KEY_TOKEN_OVERLAP_WEIGHT;
            score += countOverlap(queryTokens, toTokenSet(normalize(entry.getValue())))
                    * VALUE_TOKEN_OVERLAP_WEIGHT;
            if (score > 0) {
                candidates.add(new ScoredFact(key, entry.getValue(), score));
            }
        }
        if (candidates.isEmpty()) return Collections.emptyList();
        candidates.sort(Comparator.comparingInt((ScoredFact f) -> f.score).reversed());

        // Step 2: BERT re-ranking on top-K candidates
        int rerankCount = Math.min(SEMANTIC_RERANK_TOP_K, candidates.size());
        float[] queryEmb = encoder.encode(text);
        if (queryEmb != null) {
            float[] combinedScores = new float[rerankCount];
            for (int i = 0; i < rerankCount; i++) {
                ScoredFact f = candidates.get(i);
                float[] entryEmb = embeddingCache.get(f.key);
                if (entryEmb == null) {
                    entryEmb = encoder.encode(f.key + " " + f.value);
                    if (entryEmb != null) {
                        embeddingCache.put(f.key, entryEmb);
                    }
                }
                float sim = BertTinyEncoder.cosineSimilarity(queryEmb, entryEmb);
                combinedScores[i] = f.score + SEMANTIC_BOOST_WEIGHT * Math.max(0f, sim);
            }
            // Build index order sorted by combined score descending
            Integer[] order = new Integer[rerankCount];
            for (int i = 0; i < rerankCount; i++) order[i] = i;
            java.util.Arrays.sort(order,
                    (a, b) -> Float.compare(combinedScores[b], combinedScores[a]));
            List<String> results = new ArrayList<>(Math.min(maxFacts, rerankCount));
            for (int i = 0; i < rerankCount && results.size() < maxFacts; i++) {
                ScoredFact f = candidates.get(order[i]);
                results.add(f.key + ": " + f.value);
            }
            return results;
        }

        // BERT encode failed – fall back to keyword ranking
        List<String> results = new ArrayList<>(Math.min(maxFacts, candidates.size()));
        for (int i = 0; i < candidates.size() && results.size() < maxFacts; i++) {
            ScoredFact f = candidates.get(i);
            results.add(f.key + ": " + f.value);
        }
        return results;
    }

    public String findExactFact(String text) {
        DictionaryMatch match = findExactMatch(text);
        return match != null ? match.definition : null;
    }

    /**
     * Returns a {@link DictionaryMatch} containing both the normalized topic key and its
     * definition when the query is a recognized definition request (e.g. "what is X",
     * "define X") and an exact entry exists.  Returns {@code null} otherwise.
     */
    public DictionaryMatch findExactMatch(String text) {
        if (text == null || text.trim().isEmpty() || entries.isEmpty()) {
            return null;
        }
        String normalizedText = normalize(text);
        if (normalizedText.isEmpty()) {
            return null;
        }
        String subject = extractDefinitionSubject(normalizedText);
        if (subject == null) return null;
        String fact = entries.get(subject);
        if (fact != null) {
            return new DictionaryMatch(subject, fact);
        }
        Set<String> subjectVariants = new HashSet<>();
        subjectVariants.add(subject);
        addSingularVariants(subjectVariants, subject);
        for (String candidate : subjectVariants) {
            fact = entries.get(candidate);
            if (fact != null) {
                return new DictionaryMatch(candidate, fact);
            }
        }
        return null;
    }

    /**
     * Carries the normalised topic key and the corresponding dictionary definition
     * for a matched definition query.
     */
    public static final class DictionaryMatch {
        /** Normalized dictionary key (e.g. "photosynthesis"). */
        public final String topic;
        /** Verbatim definition text from the dictionary. */
        public final String definition;

        DictionaryMatch(String topic, String definition) {
            this.topic = topic;
            this.definition = definition;
        }
    }

    private static int countOverlap(Set<String> left, Set<String> right) {
        int overlap = 0;
        for (String token : left) {
            if (right.contains(token)) {
                overlap++;
            }
        }
        return overlap;
    }

    private static Set<String> toTokenSet(String text) {
        if (text == null || text.isEmpty()) {
            return Collections.emptySet();
        }
        String[] parts = text.split("\\s+");
        Set<String> tokens = new HashSet<>();
        for (String part : parts) {
            if (part.length() >= MIN_TOKEN_LENGTH) {
                tokens.add(part);
                addSingularVariants(tokens, part);
            }
        }
        return tokens;
    }

    private static void addSingularVariants(Set<String> tokens, String token) {
        if (token.endsWith("ies") && token.length() >= 4 && !INVARIANT_IES_WORDS.contains(token)) {
            String stem = token.substring(0, token.length() - 3);
            if (token.length() == 4) {
                tokens.add(stem + "ie");
            } else {
                tokens.add(stem + "y");
            }
            return;
        }
        if (looksLikeEsPlural(token)) {
            tokens.add(token.substring(0, token.length() - 2));
            return;
        }
        if (looksLikeSimpleSPlural(token)) {
            tokens.add(token.substring(0, token.length() - 1));
        }
    }

    private static boolean looksLikeEsPlural(String token) {
        return (token.endsWith("ses") && token.length() >= 4)
                || (token.endsWith("xes") && token.length() >= 4)
                || (token.endsWith("zes") && token.length() >= 4)
                || (token.endsWith("ches") && token.length() >= 5)
                || (token.endsWith("shes") && token.length() >= 5);
    }

    private static boolean looksLikeSimpleSPlural(String token) {
        return token.length() >= 4
                && token.endsWith("s")
                && !token.endsWith("ss")
                && !token.endsWith("us")
                && !token.endsWith("is")
                && !INVARIANT_AS_WORDS.contains(token)
                && !INVARIANT_OS_WORDS.contains(token);
    }

    private static boolean containsWholeWord(String haystack, String needle) {
        if (needle.indexOf(' ') >= 0) {
            return haystack.contains(needle);
        }
        int from = 0;
        while (true) {
            int idx = haystack.indexOf(needle, from);
            if (idx < 0) return false;
            boolean leftOk = idx == 0 || haystack.charAt(idx - 1) == ' ';
            int end = idx + needle.length();
            boolean rightOk = end == haystack.length() || haystack.charAt(end) == ' ';
            if (leftOk && rightOk) return true;
            from = idx + 1;
        }
    }

    private static String normalize(String value) {
        return value.toLowerCase(Locale.ROOT)
                .replaceAll("[^\\p{L}\\p{N}\\s]+", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static String extractDefinitionSubject(String normalizedText) {
        for (String prefix : DEFINITION_PREFIXES) {
            if (!normalizedText.startsWith(prefix)) continue;
            String subject = LEADING_ARTICLES_PATTERN
                    .matcher(normalizedText.substring(prefix.length()))
                    .replaceFirst("")
                    .trim();
            return subject.isEmpty() ? null : subject;
        }
        return null;
    }

    private static final class ScoredFact {
        final String key;
        final String value;
        final int score;

        ScoredFact(String key, String value, int score) {
            this.key = key;
            this.value = value;
            this.score = score;
        }
    }
}
