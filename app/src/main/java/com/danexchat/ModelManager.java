package com.danexchat;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Manages bundled SmolLM2 ONNX model and tokenizer assets.
 *
 * Files are stored in the app's internal files directory:
 *   <filesDir>/smollm2/model_q4.onnx
 *   <filesDir>/smollm2/tokenizer.json
 */
public class ModelManager {

    private static final String TAG = "ModelManager";

    private static final String MODEL_FILENAME     = "model_q4.onnx";
    private static final String TOKENIZER_FILENAME = "tokenizer.json";
    private static final String ASSET_MODEL_PATH = "smollm2/" + MODEL_FILENAME;
    private static final String ASSET_TOKENIZER_PATH = "smollm2/" + TOKENIZER_FILENAME;
    private static final String MODEL_DIR          = "smollm2";
    private static final long MIN_MODEL_BYTES      = 50L * 1024L * 1024L; // expected ≈90MB
    private static final long MIN_TOKENIZER_BYTES  = 1024L;

    private final File modelDir;
    private final File modelFile;
    private final File tokenizerFile;
    private final AssetManager assetManager;

    public ModelManager(Context context) {
        modelDir      = new File(context.getFilesDir(), MODEL_DIR);
        modelFile     = new File(modelDir, MODEL_FILENAME);
        tokenizerFile = new File(modelDir, TOKENIZER_FILENAME);
        assetManager = context.getAssets();
        if (!modelDir.exists()) modelDir.mkdirs();
    }

    public File getModelFile()     { return modelFile; }
    public File getTokenizerFile() { return tokenizerFile; }

    /** Returns true if both model and tokenizer files are ready in internal storage. */
    public boolean isReady() {
        ensureBundledFiles();
        return hasValidSize(modelFile, MIN_MODEL_BYTES)
                && hasValidSize(tokenizerFile, MIN_TOKENIZER_BYTES);
    }

    /**
     * Copies bundled model assets into internal storage if they're available and
     * local cached copies are missing.
     */
    public void ensureBundledFiles() {
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
