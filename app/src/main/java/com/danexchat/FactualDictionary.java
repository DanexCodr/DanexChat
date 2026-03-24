package com.danexchat;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/**
 * Lightweight local dictionary used to ground factual prompts.
 */
public class FactualDictionary {

    private static final long MAX_DICTIONARY_BYTES = 1024L * 1024L;
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
        List<String> matches = new ArrayList<>();
        for (Map.Entry<String, String> entry : entries.entrySet()) {
            String key = entry.getKey();
            if (containsWholeWord(normalizedText, key)) {
                matches.add(key + ": " + entry.getValue());
                if (matches.size() >= maxFacts) {
                    break;
                }
            }
        }
        return matches;
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
}
