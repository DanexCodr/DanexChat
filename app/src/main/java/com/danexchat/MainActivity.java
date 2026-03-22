package com.danexchat;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Main chat activity for DanexChat.
 *
 * The SmolLM2-135M-Instruct ONNX model and tokenizer are bundled with the app.
 * On startup, bundled assets are prepared in internal storage and then loaded
 * for fully on-device chat.
 */
public class MainActivity extends AppCompatActivity {
    private static final int DEFAULT_TOOLBAR_HEIGHT_DP = 56;
    private static final String TAG = "MainActivity";

    private RecyclerView  recyclerView;
    private ChatAdapter   chatAdapter;
    private List<Message> messages;

    private EditText  inputField;
    private Button    sendButton;

    private View       downloadOverlay;

    // Inline status bar (model loading / errors)
    private TextView tvStatus;
    private View inputRow;
    private ProgressBar downloadProgressBar;
    private TextView tvDownloadStatus;

    private ModelManager    modelManager;
    private SmolLMInference smolLM;

    private final ExecutorService bgExecutor = Executors.newSingleThreadExecutor();

    // -----------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        bindViews();
        setupRecyclerView();

        modelManager = new ModelManager(this);
        checkAndLoadModel();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        bgExecutor.shutdownNow();
        if (smolLM != null) smolLM.close();
    }

    // -----------------------------------------------------------------
    // View binding
    // -----------------------------------------------------------------

    private void bindViews() {
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        recyclerView      = findViewById(R.id.recyclerViewMessages);
        inputField        = findViewById(R.id.etUserInput);
        sendButton        = findViewById(R.id.btnSend);
        tvStatus          = findViewById(R.id.tvStatus);
        inputRow          = findViewById(R.id.inputRow);
        downloadOverlay   = findViewById(R.id.downloadOverlay);
        downloadProgressBar = findViewById(R.id.progressBar);
        tvDownloadStatus = findViewById(R.id.tvDownloadStatus);

        View rootLayout = findViewById(R.id.rootLayout);
        final int toolbarPaddingLeft = toolbar.getPaddingLeft();
        final int toolbarPaddingTop = toolbar.getPaddingTop();
        final int toolbarPaddingRight = toolbar.getPaddingRight();
        final int toolbarPaddingBottom = toolbar.getPaddingBottom();
        final int toolbarBaseHeight = toolbar.getLayoutParams().height;
        final int inputPaddingLeft = inputRow.getPaddingLeft();
        final int inputPaddingTop = inputRow.getPaddingTop();
        final int inputPaddingRight = inputRow.getPaddingRight();
        final int inputPaddingBottom = inputRow.getPaddingBottom();

        ViewCompat.setOnApplyWindowInsetsListener(rootLayout, (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            Insets imeInsets = insets.getInsets(WindowInsetsCompat.Type.ime());
            ViewGroup.LayoutParams toolbarLayoutParams = toolbar.getLayoutParams();
            if (toolbarLayoutParams != null) {
                int resolvedToolbarHeight = toolbarBaseHeight > 0
                        ? toolbarBaseHeight
                        : (int) (DEFAULT_TOOLBAR_HEIGHT_DP * getResources().getDisplayMetrics().density);
                toolbarLayoutParams.height = resolvedToolbarHeight + bars.top;
                toolbar.setLayoutParams(toolbarLayoutParams);
            }
            toolbar.setPadding(toolbarPaddingLeft, toolbarPaddingTop, toolbarPaddingRight, toolbarPaddingBottom);
            inputRow.setPadding(inputPaddingLeft,
                    inputPaddingTop,
                    inputPaddingRight,
                    inputPaddingBottom + Math.max(bars.bottom, imeInsets.bottom));
            return insets;
        });
        ViewCompat.requestApplyInsets(rootLayout);

        sendButton.setOnClickListener(v -> onSendClicked());
    }

    private void setupRecyclerView() {
        messages    = new ArrayList<>();
        chatAdapter = new ChatAdapter(messages);
        LinearLayoutManager llm = new LinearLayoutManager(this);
        llm.setStackFromEnd(true);
        recyclerView.setLayoutManager(llm);
        recyclerView.setAdapter(chatAdapter);
    }

    // -----------------------------------------------------------------
    // Model management
    // -----------------------------------------------------------------

    private void checkAndLoadModel() {
        if (modelManager.isReady()) {
            showDownloadOverlay(false);
            loadModelAsync();
        } else {
            startModelDownload();
        }
    }

    private void startModelDownload() {
        hideStatus();
        showDownloadOverlay(true);
        downloadProgressBar.setProgress(0);
        tvDownloadStatus.setText(getString(R.string.preparing));
        modelManager.downloadAll(new ModelManager.DownloadCallback() {
            @Override
            public void onProgress(int percent, String message) {
                downloadProgressBar.setProgress(percent);
                tvDownloadStatus.setText(message);
            }

            @Override
            public void onComplete(File modelFile, File tokenizerFile) {
                showDownloadOverlay(false);
                loadModelAsync();
            }

            @Override
            public void onError(Exception e) {
                showDownloadOverlay(false);
                showStatus(getString(R.string.download_failed));
                Log.e(TAG, "Model download failed", e);
                addAssistantMessage(getString(R.string.download_failed_chat));
                setSendEnabled(false);
            }
        });
    }

    private void loadModelAsync() {
        showStatus(getString(R.string.loading_model));
        setSendEnabled(false);

        bgExecutor.execute(() -> {
            try {
                SmolLMInference inference = new SmolLMInference(
                        modelManager.getModelFile(),
                        modelManager.getTokenizerFile());
                runOnUiThread(() -> {
                    smolLM = inference;
                    hideStatus();
                    setSendEnabled(true);
                    addAssistantMessage(getString(R.string.model_ready));
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    showStatus(getString(R.string.load_error, e.getMessage()));
                    addAssistantMessage(getString(R.string.error_prefix, e.getMessage()));
                });
            }
        });
    }

    // -----------------------------------------------------------------
    // Chat
    // -----------------------------------------------------------------

    private void onSendClicked() {
        String text = inputField.getText().toString().trim();
        if (text.isEmpty() || smolLM == null) return;

        inputField.setText("");
        setSendEnabled(false);

        addMessage(new Message(Message.ROLE_USER, text));

        // Placeholder response message updated token-by-token
        Message aiMsg = new Message(Message.ROLE_ASSISTANT, "");
        addMessage(aiMsg);

        // History snapshot (exclude the empty placeholder)
        List<Message> history = new ArrayList<>(messages.subList(0, messages.size() - 1));

        bgExecutor.execute(() ->
            smolLM.generate(history, new SmolLMInference.StreamCallback() {
                @Override
                public void onToken(String piece) {
                    runOnUiThread(() -> {
                        aiMsg.setContent(aiMsg.getContent() + piece);
                        int pos = messages.indexOf(aiMsg);
                        if (pos >= 0) chatAdapter.notifyItemChanged(pos);
                        scrollToBottom();
                    });
                }

                @Override
                public void onComplete(String fullResponse) {
                    runOnUiThread(() -> {
                        if (aiMsg.getContent().isEmpty()) {
                            aiMsg.setContent(fullResponse);
                            int pos = messages.indexOf(aiMsg);
                            if (pos >= 0) chatAdapter.notifyItemChanged(pos);
                        }
                        setSendEnabled(true);
                        scrollToBottom();
                    });
                }

                @Override
                public void onError(Exception e) {
                    runOnUiThread(() -> {
                        aiMsg.setContent(getString(R.string.error_prefix, e.getMessage()));
                        int pos = messages.indexOf(aiMsg);
                        if (pos >= 0) chatAdapter.notifyItemChanged(pos);
                        setSendEnabled(true);
                        scrollToBottom();
                    });
                }
            })
        );
    }

    // -----------------------------------------------------------------
    // UI helpers
    // -----------------------------------------------------------------

    private void addMessage(Message msg) {
        messages.add(msg);
        chatAdapter.notifyItemInserted(messages.size() - 1);
        scrollToBottom();
    }

    private void addAssistantMessage(String text) {
        addMessage(new Message(Message.ROLE_ASSISTANT, text));
    }

    private void scrollToBottom() {
        if (!messages.isEmpty()) {
            recyclerView.scrollToPosition(messages.size() - 1);
        }
    }

    private void setSendEnabled(boolean enabled) {
        sendButton.setEnabled(enabled);
        inputField.setEnabled(enabled);
    }

    private void showDownloadOverlay(boolean show) {
        downloadOverlay.setVisibility(show ? View.VISIBLE : View.GONE);
        setSendEnabled(!show);
    }

    private void showStatus(String text) {
        tvStatus.setText(text);
        tvStatus.setVisibility(View.VISIBLE);
    }

    private void hideStatus() {
        tvStatus.setVisibility(View.GONE);
    }
}
