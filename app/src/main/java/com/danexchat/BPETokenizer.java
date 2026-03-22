package com.danexchat;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Byte-level BPE tokenizer for SmolLM2.
 *
 * Parses a HuggingFace tokenizer.json and implements:
 *  - byte-to-unicode / unicode-to-byte mapping (GPT-2 style)
 *  - BPE encoding
 *  - BPE decoding
 */
public class BPETokenizer {

    // Special token IDs (SmolLM2-135M-Instruct)
    public static final long TOKEN_ENDOFTEXT   = 0L;
    public static final long TOKEN_IM_START     = 1L;
    public static final long TOKEN_IM_END       = 2L;
    public static final long TOKEN_EOS          = 0L;  // same as endoftext

    private Map<String, Long> vocab;           // token-string → id
    private Map<Long, String> reverseVocab;    // id → token-string
    private List<String[]>   merges;           // ordered list of merge pairs
    private Map<String, Integer> mergeRanks;   // "A B" → priority index

    // byte ↔ unicode maps
    private static final char[] BYTE_TO_UNICODE = new char[256];
    private static final Map<Character, Integer> UNICODE_TO_BYTE = new HashMap<>();

    static {
        buildByteMaps();
    }

    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    public BPETokenizer(File tokenizerJsonFile) throws IOException, org.json.JSONException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(tokenizerJsonFile))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line).append('\n');
        }
        parse(sb.toString());
    }

    public BPETokenizer(String tokenizerJson) throws org.json.JSONException {
        parse(tokenizerJson);
    }

    private void parse(String json) throws org.json.JSONException {
        JSONObject root = new JSONObject(json);
        JSONObject model = root.getJSONObject("model");

        // vocab
        JSONObject vocabJson = model.getJSONObject("vocab");
        vocab = new HashMap<>(vocabJson.length() * 2);
        reverseVocab = new HashMap<>(vocabJson.length() * 2);
        Iterator<String> keys = vocabJson.keys();
        while (keys.hasNext()) {
            String token = keys.next();
            long id = vocabJson.getLong(token);
            vocab.put(token, id);
            reverseVocab.put(id, token);
        }

        // merges
        JSONArray mergesJson = model.getJSONArray("merges");
        merges = new ArrayList<>(mergesJson.length());
        mergeRanks = new HashMap<>(mergesJson.length() * 2);
        for (int i = 0; i < mergesJson.length(); i++) {
            String merge = mergesJson.getString(i);
            int spaceIdx = merge.indexOf(' ');
            if (spaceIdx < 0) continue;
            String left  = merge.substring(0, spaceIdx);
            String right = merge.substring(spaceIdx + 1);
            merges.add(new String[]{left, right});
            mergeRanks.put(merge, i);
        }

        // Load added_tokens (special tokens) from tokenizer.json if present
        if (root.has("added_tokens")) {
            JSONArray addedTokens = root.getJSONArray("added_tokens");
            for (int i = 0; i < addedTokens.length(); i++) {
                JSONObject t = addedTokens.getJSONObject(i);
                String content = t.getString("content");
                long id = t.getLong("id");
                vocab.put(content, id);
                reverseVocab.put(id, content);
            }
        }
    }

    // -----------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------

    /** Encode a plain text string to a list of token IDs. */
    public long[] encode(String text) {
        if (text == null || text.isEmpty()) return new long[0];

        // Split into words (ByteLevel pre-tokenizer like GPT-2)
        // Words are split on whitespace; leading space is preserved on words
        // except the first.
        List<String> words = splitWords(text);
        List<Long> ids = new ArrayList<>();

        for (String word : words) {
            // Map bytes → unicode chars
            byte[] bytes;
            try {
                bytes = word.getBytes(StandardCharsets.UTF_8);
            } catch (Exception e) {
                continue;
            }
            StringBuilder unicodeWord = new StringBuilder();
            for (byte b : bytes) {
                unicodeWord.append(BYTE_TO_UNICODE[b & 0xFF]);
            }
            // BPE on the unicode characters
            List<Long> wordIds = bpe(unicodeWord.toString());
            ids.addAll(wordIds);
        }

        long[] result = new long[ids.size()];
        for (int i = 0; i < ids.size(); i++) result[i] = ids.get(i);
        return result;
    }

    /**
     * Encode text that may contain special tokens like <|im_start|>.
     * Special tokens are handled directly without BPE.
     */
    public long[] encodeWithSpecialTokens(String text) {
        // Extract known special tokens from vocab
        List<String> specialTokens = new ArrayList<>();
        for (String token : vocab.keySet()) {
            if (token.startsWith("<|") && token.endsWith("|>")) {
                specialTokens.add(token);
            }
        }
        // Sort by length descending so longer tokens match first
        specialTokens.sort((a, b) -> b.length() - a.length());

        List<Long> ids = new ArrayList<>();
        int pos = 0;
        while (pos < text.length()) {
            boolean matched = false;
            for (String st : specialTokens) {
                if (text.startsWith(st, pos)) {
                    Long id = vocab.get(st);
                    if (id != null) ids.add(id);
                    pos += st.length();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                // Find next special token position
                int nextSpecial = text.length();
                for (String st : specialTokens) {
                    int idx = text.indexOf(st, pos);
                    if (idx >= 0 && idx < nextSpecial) nextSpecial = idx;
                }
                // Encode the regular text chunk
                String chunk = text.substring(pos, nextSpecial);
                long[] chunkIds = encode(chunk);
                for (long id : chunkIds) ids.add(id);
                pos = nextSpecial;
            }
        }

        long[] result = new long[ids.size()];
        for (int i = 0; i < ids.size(); i++) result[i] = ids.get(i);
        return result;
    }

    // -----------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------

    /** Decode a list of token IDs back to a UTF-8 string. */
    public String decode(List<Long> ids) {
        return decode(ids.stream().mapToLong(Long::longValue).toArray());
    }

    public String decode(long[] ids) {
        if (ids == null || ids.length == 0) return "";

        StringBuilder sb = new StringBuilder();
        for (long id : ids) {
            String token = reverseVocab.get(id);
            if (token == null) continue;
            sb.append(token);
        }

        // Convert unicode string back to bytes
        String unicodeStr = sb.toString();
        byte[] bytes = new byte[unicodeStr.length()];
        int len = 0;
        for (int i = 0; i < unicodeStr.length(); i++) {
            char c = unicodeStr.charAt(i);
            Integer b = UNICODE_TO_BYTE.get(c);
            if (b != null) {
                bytes[len++] = b.byteValue();
            }
            // Unknown chars are skipped (special tokens handled above)
        }
        return new String(bytes, 0, len, StandardCharsets.UTF_8);
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    /**
     * Apply BPE merges to a single pre-tokenized word.
     * @param word Unicode-encoded word string (each char represents one byte)
     * @return list of token IDs
     */
    private List<Long> bpe(String word) {
        // Initialize with individual characters
        LinkedList<String> pieces = new LinkedList<>();
        for (int i = 0; i < word.length(); ) {
            int cp = word.codePointAt(i);
            pieces.add(new String(Character.toChars(cp)));
            i += Character.charCount(cp);
        }

        // Iteratively apply the best merge
        while (pieces.size() > 1) {
            // Find the pair with the lowest merge rank
            int bestRank = Integer.MAX_VALUE;
            String bestLeft = null, bestRight = null;

            String prev = null;
            for (String piece : pieces) {
                if (prev != null) {
                    String pairKey = prev + " " + piece;
                    Integer rank = mergeRanks.get(pairKey);
                    if (rank != null && rank < bestRank) {
                        bestRank = rank;
                        bestLeft  = prev;
                        bestRight = piece;
                    }
                }
                prev = piece;
            }

            if (bestLeft == null) break; // No more merges possible

            // Apply the merge: combine all consecutive occurrences
            String merged = bestLeft + bestRight;
            LinkedList<String> next = new LinkedList<>();
            Iterator<String> it = pieces.iterator();
            String current = it.next();
            while (it.hasNext()) {
                String nxt = it.next();
                if (current.equals(bestLeft) && nxt.equals(bestRight)) {
                    next.add(merged);
                    if (it.hasNext()) {
                        current = it.next();
                    } else {
                        current = null;
                    }
                } else {
                    next.add(current);
                    current = nxt;
                }
            }
            if (current != null) next.add(current);
            pieces = next;
        }

        // Map pieces → IDs
        List<Long> ids = new ArrayList<>(pieces.size());
        for (String piece : pieces) {
            Long id = vocab.get(piece);
            if (id != null) {
                ids.add(id);
            }
        }
        return ids;
    }

    /**
     * Split text into words the same way GPT-2's ByteLevel pre-tokenizer does.
     * Words are separated by spaces; each non-first word gets a leading 'Ġ'.
     */
    private static List<String> splitWords(String text) {
        List<String> words = new ArrayList<>();
        // Regex: match whitespace-prefixed or leading word chunks
        int i = 0;
        boolean firstWord = true;
        while (i < text.length()) {
            // Consume leading spaces and attach to the next word
            int spaceStart = i;
            while (i < text.length() && text.charAt(i) == ' ') i++;
            int wordStart = i;
            while (i < text.length() && text.charAt(i) != ' ') i++;
            int wordEnd = i;

            if (wordStart < wordEnd) {
                String word = text.substring(wordStart, wordEnd);
                if (!firstWord || spaceStart < wordStart) {
                    // prefix space
                    word = " " + word;
                }
                words.add(word);
                firstWord = false;
            }
        }
        if (words.isEmpty() && !text.isEmpty()) {
            words.add(text);
        }
        return words;
    }

    // -----------------------------------------------------------------
    // Byte ↔ unicode maps (GPT-2 style)
    // -----------------------------------------------------------------

    private static void buildByteMaps() {
        // Bytes that map to themselves (printable ASCII + Latin supplement)
        List<Integer> bs = new ArrayList<>();
        for (int b = '!'; b <= '~'; b++) bs.add(b);
        for (int b = 0xA1; b <= 0xAC; b++) bs.add(b);
        for (int b = 0xAE; b <= 0xFF; b++) bs.add(b);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }

        for (int i = 0; i < bs.size(); i++) {
            int byteVal  = bs.get(i);
            char unichar = (char) (int) cs.get(i);
            BYTE_TO_UNICODE[byteVal] = unichar;
            UNICODE_TO_BYTE.put(unichar, byteVal);
        }
    }
}
