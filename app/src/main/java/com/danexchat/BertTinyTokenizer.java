package com.danexchat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Minimal WordPiece tokenizer for BERT-tiny (uncased).
 *
 * Handles basic text normalization, punctuation splitting, and WordPiece
 * subword tokenization. Produces [CLS]/[SEP]-delimited token-ID sequences
 * for BERT model input.
 */
public class BertTinyTokenizer {

    private static final String TOKEN_UNK = "[UNK]";
    private static final String WORDPIECE_PREFIX = "##";
    private static final int MAX_CHARS_PER_WORD = 100;

    // Fixed IDs for standard BERT uncased vocabulary
    static final int TOKEN_ID_PAD = 0;
    static final int TOKEN_ID_UNK = 100;
    static final int TOKEN_ID_CLS = 101;
    static final int TOKEN_ID_SEP = 102;

    private final Map<String, Integer> vocab;

    public BertTinyTokenizer(File vocabFile) throws IOException {
        vocab = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(vocabFile))) {
            String line;
            int id = 0;
            while ((line = reader.readLine()) != null) {
                String token = line.trim();
                if (!token.isEmpty()) {
                    vocab.put(token, id);
                }
                id++;
            }
        }
    }

    /**
     * Encode {@code text} to a token-ID array with [CLS] at index 0 and
     * [SEP] at the end. Truncated to at most {@code maxLen} tokens total.
     */
    public long[] encode(String text, int maxLen) {
        if (text == null || text.trim().isEmpty() || maxLen < 2) {
            return new long[]{TOKEN_ID_CLS, TOKEN_ID_SEP};
        }
        List<String> tokens = tokenize(text);
        int contentLen = Math.min(tokens.size(), maxLen - 2);
        long[] ids = new long[contentLen + 2];
        ids[0] = TOKEN_ID_CLS;
        for (int i = 0; i < contentLen; i++) {
            ids[i + 1] = vocab.getOrDefault(tokens.get(i), TOKEN_ID_UNK);
        }
        ids[contentLen + 1] = TOKEN_ID_SEP;
        return ids;
    }

    private List<String> tokenize(String text) {
        String normalized = text.toLowerCase(Locale.ROOT).trim();
        if (normalized.isEmpty()) return Collections.emptyList();
        List<String> basicTokens = basicTokenize(normalized);
        List<String> subTokens = new ArrayList<>();
        for (String token : basicTokens) {
            subTokens.addAll(wordpiece(token));
        }
        return subTokens;
    }

    private static List<String> basicTokenize(String text) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (isPunctuation(ch)) {
                if (current.length() > 0) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }
                tokens.add(String.valueOf(ch));
            } else if (Character.isWhitespace(ch)) {
                if (current.length() > 0) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }
            } else {
                current.append(ch);
            }
        }
        if (current.length() > 0) {
            tokens.add(current.toString());
        }
        return tokens;
    }

    private List<String> wordpiece(String word) {
        if (word.length() > MAX_CHARS_PER_WORD) {
            return Collections.singletonList(TOKEN_UNK);
        }
        List<String> subTokens = new ArrayList<>();
        int start = 0;
        while (start < word.length()) {
            int end = word.length();
            String found = null;
            while (start < end) {
                String sub = word.substring(start, end);
                String candidate = start > 0 ? WORDPIECE_PREFIX + sub : sub;
                if (vocab.containsKey(candidate)) {
                    found = candidate;
                    break;
                }
                end--;
            }
            if (found == null) {
                return Collections.singletonList(TOKEN_UNK);
            }
            subTokens.add(found);
            start = end;
        }
        return subTokens;
    }

    private static boolean isPunctuation(char ch) {
        return (ch >= '!' && ch <= '/')
                || (ch >= ':' && ch <= '@')
                || (ch >= '[' && ch <= '`')
                || (ch >= '{' && ch <= '~');
    }
}
