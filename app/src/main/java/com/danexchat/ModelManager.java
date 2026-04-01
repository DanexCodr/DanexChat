package com.danexchat;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Manages bundled SmolLM2 ONNX model and tokenizer assets, and the optional
 * BERT-tiny model used for semantic intent routing and fact retrieval.
 *
 * Files are stored in the app's internal files directory:
 *   <filesDir>/smollm2/model_q4.onnx
 *   <filesDir>/smollm2/tokenizer.json
 *   <filesDir>/smollm2/wordnet.json
 *   <filesDir>/bert_tiny/bert_tiny.onnx   (optional)
 *   <filesDir>/bert_tiny/bert_vocab.txt   (optional)
 */
public class ModelManager {

    private static final String TAG = "ModelManager";

    private static final String MODEL_FILENAME      = "model_q4.onnx";
    private static final String TOKENIZER_FILENAME  = "tokenizer.json";
    private static final String DICTIONARY_FILENAME = "wordnet.json";
    private static final String ASSET_MODEL_PATH      = "smollm2/" + MODEL_FILENAME;
    private static final String ASSET_TOKENIZER_PATH  = "smollm2/" + TOKENIZER_FILENAME;
    private static final String ASSET_DICTIONARY_PATH = "smollm2/" + DICTIONARY_FILENAME;
    private static final String MODEL_DIR           = "smollm2";
    private static final long MIN_MODEL_BYTES       = 50L * 1024L * 1024L; // expected ≈90MB
    private static final long MIN_TOKENIZER_BYTES   = 1024L;
    private static final long MIN_DICTIONARY_BYTES  = 32L;

    // BERT-tiny assets (optional – gracefully absent when not bundled)
    private static final String BERT_DIR              = "bert_tiny";
    private static final String BERT_MODEL_FILENAME   = "bert_tiny.onnx";
    private static final String BERT_VOCAB_FILENAME   = "bert_vocab.txt";
    private static final String ASSET_BERT_MODEL_PATH = BERT_DIR + "/" + BERT_MODEL_FILENAME;
    private static final String ASSET_BERT_VOCAB_PATH = BERT_DIR + "/" + BERT_VOCAB_FILENAME;
    private static final long MIN_BERT_MODEL_BYTES    = 1024L;
    private static final long MIN_BERT_VOCAB_BYTES    = 1024L;

    private final File modelDir;
    private final File modelFile;
    private final File tokenizerFile;
    private final File dictionaryFile;
    private final File bertDir;
    private final File bertModelFile;
    private final File bertVocabFile;
    private final AssetManager assetManager;

    public ModelManager(Context context) {
        modelDir       = new File(context.getFilesDir(), MODEL_DIR);
        modelFile      = new File(modelDir, MODEL_FILENAME);
        tokenizerFile  = new File(modelDir, TOKENIZER_FILENAME);
        dictionaryFile = new File(modelDir, DICTIONARY_FILENAME);
        bertDir        = new File(context.getFilesDir(), BERT_DIR);
        bertModelFile  = new File(bertDir, BERT_MODEL_FILENAME);
        bertVocabFile  = new File(bertDir, BERT_VOCAB_FILENAME);
        assetManager   = context.getAssets();
        if (!modelDir.exists()) modelDir.mkdirs();
        if (!bertDir.exists())  bertDir.mkdirs();
    }

    public File getModelFile()      { return modelFile; }
    public File getTokenizerFile()  { return tokenizerFile; }
    public File getDictionaryFile() { return dictionaryFile; }
    public File getBertModelFile()  { return bertModelFile; }
    public File getBertVocabFile()  { return bertVocabFile; }

    /** Returns true if bundled assets can be prepared into internal storage and pass size checks. */
    public boolean isReady() {
        ensureBundledFiles();
        return hasValidSize(modelFile, MIN_MODEL_BYTES)
                && hasValidSize(tokenizerFile, MIN_TOKENIZER_BYTES)
                && hasValidSize(dictionaryFile, MIN_DICTIONARY_BYTES);
    }

    /**
     * Returns true when both BERT-tiny assets (model + vocab) are available in
     * internal storage. These assets are optional; the app works without them.
     */
    public boolean hasBertFiles() {
        ensureBertFiles();
        return hasValidSize(bertModelFile, MIN_BERT_MODEL_BYTES)
                && hasValidSize(bertVocabFile, MIN_BERT_VOCAB_BYTES);
    }

    /**
     * Copies bundled BERT-tiny assets into internal storage if they are present
     * in the APK and the local cached copies are missing or stale.
     */
    public void ensureBertFiles() {
        if (!hasValidSize(bertVocabFile, MIN_BERT_VOCAB_BYTES)) {
            try {
                copyAssetIfExists(ASSET_BERT_VOCAB_PATH, bertVocabFile);
            } catch (IOException e) {
                Log.w(TAG, "BERT vocab asset not available", e);
            }
        }
        if (!hasValidSize(bertModelFile, MIN_BERT_MODEL_BYTES)) {
            try {
                copyAssetIfExists(ASSET_BERT_MODEL_PATH, bertModelFile);
            } catch (IOException e) {
                Log.w(TAG, "BERT model asset not available", e);
            }
        }
    }

    /**
     * Copies bundled model assets into internal storage if they're available and
     * local cached copies are missing.
     */
    public void ensureBundledFiles() {
        if (!hasValidSize(dictionaryFile, MIN_DICTIONARY_BYTES)) {
            try {
                copyAssetIfExists(ASSET_DICTIONARY_PATH, dictionaryFile);
            } catch (IOException e) {
                Log.w(TAG, "Bundled dictionary asset was not available", e);
            }
        }
        if (!hasValidSize(tokenizerFile, MIN_TOKENIZER_BYTES)) {
            try {
                copyAssetIfExists(ASSET_TOKENIZER_PATH, tokenizerFile);
            } catch (IOException e) {
                Log.w(TAG, "Bundled tokenizer asset was not available", e);
            }
        }
        if (!hasValidSize(modelFile, MIN_MODEL_BYTES)) {
            try {
                copyAssetIfExists(ASSET_MODEL_PATH, modelFile);
            } catch (IOException e) {
                Log.w(TAG, "Bundled model asset was not available", e);
            }
        }
    }

    private boolean hasValidSize(File file, long minBytes) {
        return file.exists() && file.length() >= minBytes;
    }

    private void copyAssetIfExists(String assetPath, File dest) throws IOException {
        File parent = dest.getParentFile();
        if (parent == null) {
            throw new IOException("Destination has no parent directory: " + dest.getAbsolutePath());
        }
        File tmp = new File(parent, dest.getName() + ".tmp");
        if (tmp.exists() && !tmp.delete()) {
            Log.w(TAG, "Failed to clean old temp file before asset copy: " + tmp.getAbsolutePath());
        }
        try {
            try (InputStream in = assetManager.open(assetPath);
                 FileOutputStream out = new FileOutputStream(tmp)) {
                byte[] buffer = new byte[8192];
                int count;
                while ((count = in.read(buffer)) != -1) {
                    out.write(buffer, 0, count);
                }
            }

            if (!tmp.renameTo(dest)) {
                java.nio.file.Files.move(tmp.toPath(), dest.toPath(),
                        java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException e) {
            if (!tmp.delete()) {
                Log.w(TAG, "Failed to delete temp asset copy: " + tmp.getAbsolutePath());
            }
            throw e;
        }
    }

}
