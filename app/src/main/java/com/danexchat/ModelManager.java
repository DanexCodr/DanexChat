package com.danexchat;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Manages downloading and caching of the SmolLM2 ONNX model and tokenizer.
 *
 * Files are stored in the app's internal files directory:
 *   <filesDir>/smollm2/model_q4.onnx
 *   <filesDir>/smollm2/tokenizer.json
 */
public class ModelManager {

    private static final String TAG = "ModelManager";

    // HuggingFace model URLs for SmolLM2-135M-Instruct
    private static final String MODEL_URL =
            "https://huggingface.co/onnx-community/SmolLM2-135M-Instruct/resolve/main/onnx/model_q4.onnx";
    private static final String TOKENIZER_URL =
            "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json";

    private static final String MODEL_FILENAME     = "model_q4.onnx";
    private static final String TOKENIZER_FILENAME = "tokenizer.json";
    private static final String MODEL_DIR          = "smollm2";

    private final File modelDir;
    private final File modelFile;
    private final File tokenizerFile;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    public interface DownloadCallback {
        void onProgress(int percent, String message);
        void onComplete(File modelFile, File tokenizerFile);
        void onError(Exception e);
    }

    public ModelManager(Context context) {
        modelDir      = new File(context.getFilesDir(), MODEL_DIR);
        modelFile     = new File(modelDir, MODEL_FILENAME);
        tokenizerFile = new File(modelDir, TOKENIZER_FILENAME);
        if (!modelDir.exists()) modelDir.mkdirs();
    }

    public File getModelFile()     { return modelFile; }
    public File getTokenizerFile() { return tokenizerFile; }

    /** Returns true if both model and tokenizer files are already downloaded. */
    public boolean isReady() {
        return modelFile.exists() && modelFile.length() > 0
            && tokenizerFile.exists() && tokenizerFile.length() > 0;
    }

    /**
     * Download both model and tokenizer files in a background thread.
     * Progress and completion are reported on the main thread via {@code callback}.
     */
    public void downloadAll(DownloadCallback callback) {
        new Thread(() -> {
            try {
                // Step 1: tokenizer (~1-2 MB, fast)
                if (!tokenizerFile.exists() || tokenizerFile.length() == 0) {
                    postProgress(callback, 0, "Downloading tokenizer…");
                    downloadFile(TOKENIZER_URL, tokenizerFile, (p) ->
                            postProgress(callback, p / 20, "Downloading tokenizer…"));
                }

                // Step 2: model (large – ~90 MB for q4)
                if (!modelFile.exists() || modelFile.length() == 0) {
                    postProgress(callback, 5, "Downloading SmolLM2-135M model (≈90 MB)…");
                    downloadFile(MODEL_URL, modelFile, (p) ->
                            postProgress(callback, 5 + (int)(p * 0.95), "Downloading model… " + p + "%"));
                }

                mainHandler.post(() -> callback.onComplete(modelFile, tokenizerFile));
            } catch (Exception e) {
                Log.e(TAG, "Download failed", e);
                // Clean up partial files
                if (modelFile.exists() && modelFile.length() == 0) modelFile.delete();
                mainHandler.post(() -> callback.onError(e));
            }
        }, "ModelDownload").start();
    }

    // -----------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------

    private interface ProgressListener {
        void onProgress(int percent);
    }

    private void downloadFile(String urlStr, File dest, ProgressListener progress)
            throws IOException {

        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setInstanceFollowRedirects(true);
        conn.setConnectTimeout(30_000);
        conn.setReadTimeout(60_000);
        conn.connect();

        // Follow redirects (HuggingFace CDN)
        int code = conn.getResponseCode();
        while (code == HttpURLConnection.HTTP_MOVED_PERM
                || code == HttpURLConnection.HTTP_MOVED_TEMP
                || code == 307 || code == 308) {
            String location = conn.getHeaderField("Location");
            conn.disconnect();
            url  = new URL(location);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(30_000);
            conn.setReadTimeout(60_000);
            conn.connect();
            code = conn.getResponseCode();
        }

        if (code != HttpURLConnection.HTTP_OK) {
            conn.disconnect();
            throw new IOException("HTTP " + code + " fetching " + urlStr);
        }

        long total = conn.getContentLengthLong();
        long downloaded = 0;
        byte[] buf = new byte[65536];

        File tmp = new File(dest.getParent(), dest.getName() + ".tmp");
        try (InputStream in       = conn.getInputStream();
             FileOutputStream out = new FileOutputStream(tmp)) {
            int n;
            while ((n = in.read(buf)) >= 0) {
                out.write(buf, 0, n);
                downloaded += n;
                if (total > 0) {
                    progress.onProgress((int)(downloaded * 100 / total));
                }
            }
        } finally {
            conn.disconnect();
        }

        if (!tmp.renameTo(dest)) {
            // Fallback: copy then delete
            java.nio.file.Files.move(tmp.toPath(), dest.toPath(),
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
        Log.i(TAG, "Downloaded " + dest.getName() + " (" + downloaded / 1024 + " KB)");
    }

    private void postProgress(DownloadCallback cb, int percent, String message) {
        mainHandler.post(() -> cb.onProgress(percent, message));
    }
}
