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

/**
 * Lightweight local dictionary used to ground factual prompts.
 */
public class FactualDictionary {

    private static final long MAX_DICTIONARY_BYTES = 1024L * 1024L;
    private static final int WHOLE_KEY_MATCH_WEIGHT = 4;
    private static final int KEY_TOKEN_OVERLAP_WEIGHT = 3;
    private static final int VALUE_TOKEN_OVERLAP_WEIGHT = 1;
    private static final int MIN_TOKEN_LENGTH = 2;
    private final Map<String, String> entries;

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
        if (token.endsWith("ies") && token.length() >= 4) {
            tokens.add(token.substring(0, token.length() - 3) + "y");
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
        return token.length() >= 4
                && (token.endsWith("ses")
                || token.endsWith("xes")
                || token.endsWith("zes")
                || token.endsWith("ches")
                || token.endsWith("shes"));
    }

    private static boolean looksLikeSimpleSPlural(String token) {
        return token.length() >= 3
                && token.endsWith("s")
                && !token.endsWith("ss")
                && !token.endsWith("us")
                && !token.endsWith("is");
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
