package com.danexchat;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

/**
 * Main chat activity for DanexChat.
 *
 * The SmolLM2-135M-Instruct ONNX model and tokenizer are bundled with the app.
 * On startup, bundled assets are prepared in internal storage and then loaded
 * for fully on-device chat.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int DEFAULT_TOOLBAR_HEIGHT_DP = 56;
    // Topic overlap is computed with Jaccard similarity; below this value we treat
    // consecutive definition-style prompts as a topic switch and reset model context.
    // Lower values make resets more aggressive, while higher values preserve context
    // unless prompts are clearly different. 0.2 keeps mildly related follow-ups.
    private static final float TOPIC_TOKEN_OVERLAP_THRESHOLD = 0.2f;
    private static final float ARS_RESOLUTION_CONFIDENCE_THRESHOLD = 0.65f;
    private static final float ARS_RESOLUTION_CONFIDENCE_NONE = 0f;
    private static final float ARS_RESOLUTION_CONFIDENCE_NOT_APPLICABLE = 1f;
    private static final float ARS_RESOLUTION_CONFIDENCE_SINGLE_SUBJECT = 0.95f;
    private static final float ARS_RESOLUTION_CONFIDENCE_MULTI_SUBJECT = 0.55f;
    private static final int MIN_TOPIC_TOKEN_LENGTH = 3;
    private static final Pattern TOPIC_TOKEN_SPLIT_PATTERN = Pattern.compile("[^\\p{L}\\p{N}]+");
    private static final Pattern DEFINITION_ARTICLE_PATTERN = Pattern.compile("^(a|an|the)\\s+");
    private static final Set<String> AMBIGUOUS_REFERENCES = new HashSet<>(Arrays.asList(
            "it", "this", "that", "they", "them", "he", "she", "him", "her", "these", "those"
    ));
    private static final Set<String> TOPIC_STOPWORDS = new HashSet<>(Arrays.asList(
            "a", "an", "the", "is", "are", "was", "were", "am", "be", "to", "of", "for", "in",
            "on", "at", "and", "or", "but", "with", "about", "this", "that", "it", "its", "as",
            "what", "who", "when", "where", "why", "how", "tell", "me", "please",
            "do", "does", "can", "will", "would"
    ));

    private RecyclerView  recyclerView;
    private ChatAdapter   chatAdapter;
    private List<Message> messages;
    private List<Message> conversationHistory;

    private EditText  inputField;
    private Button    sendButton;

    // Inline status bar (model loading / errors)
    private TextView tvStatus;
    private View inputRow;

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
        messages = new ArrayList<>();
        conversationHistory = new ArrayList<>();
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
            loadModelAsync();
        } else {
            showStatus(getString(R.string.bundled_model_missing));
            addAssistantMessage(getString(R.string.bundled_model_missing_chat));
            setSendEnabled(false);
        }
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

        if (shouldResetConversationContext(text)) {
            Log.d(TAG, "Resetting model conversation context for detected topic switch");
            conversationHistory.clear();
        }

        ArsDecision arsDecision = resolveAmbiguityAndSpecifier(text);
        inputField.setText("");

        Message userMsg = new Message(Message.ROLE_USER, text);
        addMessage(userMsg);
        conversationHistory.add(userMsg);
        if (arsDecision.clarifyingQuestion != null) {
            Message clarification = new Message(Message.ROLE_ASSISTANT, arsDecision.clarifyingQuestion);
            addMessage(clarification);
            conversationHistory.add(clarification);
            return;
        }
        if (!arsDecision.modelText.equals(text)) {
            Log.d(TAG, "ARS specifier applied: " + arsDecision.modelText);
        }
        setSendEnabled(false);

        // Placeholder response message updated token-by-token
        Message aiMsg = new Message(Message.ROLE_ASSISTANT, "");
        addMessage(aiMsg);
        List<Message> history = new ArrayList<>(conversationHistory);
        if (!arsDecision.modelText.equals(text)) {
            history.set(history.size() - 1, new Message(Message.ROLE_USER, arsDecision.modelText));
        }

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
                        if (!fullResponse.equals(aiMsg.getContent())) {
                            aiMsg.setContent(fullResponse);
                            int pos = messages.indexOf(aiMsg);
                            if (pos >= 0) chatAdapter.notifyItemChanged(pos);
                        }
                        if (conversationHistory.isEmpty()
                                || conversationHistory.get(conversationHistory.size() - 1).isUser()) {
                            conversationHistory.add(new Message(Message.ROLE_ASSISTANT, aiMsg.getContent()));
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

    private void showStatus(String text) {
        tvStatus.setText(text);
        tvStatus.setVisibility(View.VISIBLE);
    }

    private void hideStatus() {
        tvStatus.setVisibility(View.GONE);
    }

    private boolean shouldResetConversationContext(String newUserText) {
        Message lastUserMessage = findLastUserMessage();
        if (lastUserMessage == null) return false;

        if (!isDefinitionStyleQuestion(lastUserMessage.getContent())
                || !isDefinitionStyleQuestion(newUserText)) {
            return false;
        }

        Set<String> previousTopicTokens = extractTopicTokens(lastUserMessage.getContent());
        Set<String> newTopicTokens = extractTopicTokens(newUserText);
        if (previousTopicTokens.isEmpty() || newTopicTokens.isEmpty()) {
            return false;
        }

        Set<String> overlap = new HashSet<>(previousTopicTokens);
        overlap.retainAll(newTopicTokens);
        Set<String> union = new HashSet<>(previousTopicTokens);
        union.addAll(newTopicTokens);
        float overlapRatio = (float) overlap.size() / union.size();
        return overlapRatio < TOPIC_TOKEN_OVERLAP_THRESHOLD;
    }

    private Message findLastUserMessage() {
        for (int i = conversationHistory.size() - 1; i >= 0; i--) {
            Message message = conversationHistory.get(i);
            if (message.isUser()) {
                return message;
            }
        }
        return null;
    }

    private static boolean isDefinitionStyleQuestion(String text) {
        String normalized = text.trim().toLowerCase(Locale.ROOT);
        if (normalized.isEmpty()) return false;
        if (normalized.endsWith("?")) {
            normalized = normalized.substring(0, normalized.length() - 1).trim();
        }
        return normalized.startsWith("what is ")
                || normalized.startsWith("what are ")
                || normalized.startsWith("who is ")
                || normalized.startsWith("who are ")
                || normalized.startsWith("define ")
                || normalized.startsWith("tell me about ");
    }

    private static Set<String> extractTopicTokens(String text) {
        Set<String> tokens = new HashSet<>();
        String[] parts = TOPIC_TOKEN_SPLIT_PATTERN.split(text.toLowerCase(Locale.ROOT));
        for (String part : parts) {
            if (part.length() < MIN_TOPIC_TOKEN_LENGTH || TOPIC_STOPWORDS.contains(part)) continue;
            tokens.add(part);
        }
        return tokens;
    }

    private ArsDecision resolveAmbiguityAndSpecifier(String userText) {
        String subject = extractDefinitionSubject(userText);
        if (subject == null) {
            if (!containsAmbiguousReference(userText)) {
                // No ARS disambiguation needed for this input.
                return new ArsDecision(userText, null, ARS_RESOLUTION_CONFIDENCE_NOT_APPLICABLE);
            }
            SubjectResolution resolution = resolveMostRecentConcreteSubject();
            if (resolution == null || resolution.confidence < ARS_RESOLUTION_CONFIDENCE_THRESHOLD) {
                return new ArsDecision(userText, getString(R.string.ars_clarify_ambiguous_reference),
                        resolution == null ? ARS_RESOLUTION_CONFIDENCE_NONE : resolution.confidence);
            }
            String rewritten = rewriteAmbiguousFollowUp(userText, sanitizeResolvedSubject(resolution.subject));
            return new ArsDecision(rewritten, null, resolution.confidence);
        }
        if (!AMBIGUOUS_REFERENCES.contains(subject)) {
            return new ArsDecision(userText, null, ARS_RESOLUTION_CONFIDENCE_NOT_APPLICABLE);
        }

        SubjectResolution resolution = resolveMostRecentConcreteSubject();
        if (resolution == null || resolution.confidence < ARS_RESOLUTION_CONFIDENCE_THRESHOLD) {
            return new ArsDecision(userText, getString(R.string.ars_clarify_ambiguous_reference),
                    resolution == null ? ARS_RESOLUTION_CONFIDENCE_NONE : resolution.confidence);
        }
        return new ArsDecision(
                rewriteDefinitionSubject(userText, sanitizeResolvedSubject(resolution.subject)),
                null,
                resolution.confidence);
    }

    private static String extractDefinitionSubject(String text) {
        String normalized = text.trim().toLowerCase(Locale.ROOT);
        if (normalized.endsWith("?")) {
            normalized = normalized.substring(0, normalized.length() - 1).trim();
        }
        String[] prefixes = {"what is ", "what are ", "who is ", "who are ", "define ", "tell me about "};
        for (String prefix : prefixes) {
            if (!normalized.startsWith(prefix)) continue;
            String subject = normalized.substring(prefix.length()).trim();
            subject = DEFINITION_ARTICLE_PATTERN.matcher(subject).replaceFirst("").trim();
            return subject.isEmpty() ? null : subject;
        }
        return null;
    }

    private static String rewriteDefinitionSubject(String originalText, String subject) {
        String trimmed = originalText.trim();
        boolean hasQuestionMark = trimmed.endsWith("?");
        String withoutQuestionMark = hasQuestionMark
                ? trimmed.substring(0, trimmed.length() - 1).trim()
                : trimmed;
        String normalized = withoutQuestionMark.toLowerCase(Locale.ROOT);
        String[] prefixes = {"what is ", "what are ", "who is ", "who are ", "define ", "tell me about "};
        for (String prefix : prefixes) {
            if (!normalized.startsWith(prefix)) continue;
            String rewritten = withoutQuestionMark.substring(0, prefix.length()) + subject;
            return hasQuestionMark ? rewritten + "?" : rewritten;
        }
        return "what is " + subject + "?";
    }

    private SubjectResolution resolveMostRecentConcreteSubject() {
        String mostRecentSubject = null;
        Set<String> uniqueSubjects = new HashSet<>();
        for (int i = conversationHistory.size() - 1; i >= 0; i--) {
            Message message = conversationHistory.get(i);
            if (!message.isUser()) continue;
            String subject = extractDefinitionSubject(message.getContent());
            if (subject == null || AMBIGUOUS_REFERENCES.contains(subject)) continue;
            if (mostRecentSubject == null) {
                mostRecentSubject = subject;
            }
            uniqueSubjects.add(subject);
        }
        if (mostRecentSubject == null) return null;
        // uniqueSubjects cannot be empty here because we always add the same concrete
        // subject that initializes mostRecentSubject in the loop above.
        // Two-level confidence scoring: one consistent recent concrete subject is highly
        // reliable, while multiple distinct recent subjects mean the anaphora target is
        // likely unclear.
        float confidence = uniqueSubjects.size() == 1
                ? ARS_RESOLUTION_CONFIDENCE_SINGLE_SUBJECT
                : ARS_RESOLUTION_CONFIDENCE_MULTI_SUBJECT;
        return new SubjectResolution(mostRecentSubject, confidence);
    }

    private static boolean containsAmbiguousReference(String text) {
        String[] parts = TOPIC_TOKEN_SPLIT_PATTERN.split(text.toLowerCase(Locale.ROOT));
        for (String part : parts) {
            if (AMBIGUOUS_REFERENCES.contains(part)) return true;
        }
        return false;
    }

    private static String rewriteAmbiguousFollowUp(String originalText, String subject) {
        String lower = originalText.toLowerCase(Locale.ROOT);
        if (lower.contains("what about")) {
            return "tell me more about " + subject;
        }
        // Last-resort fallback for non-definition ambiguous follow-ups; this favors
        // preserving user wording with explicit context over more invasive rephrasing.
        return originalText + " (about " + subject + ")";
    }

    private static String sanitizeResolvedSubject(String subject) {
        return subject.replace('\n', ' ')
                .replace('\r', ' ')
                .replace('\t', ' ')
                .trim();
    }

    /** Result of ARS preprocessing before model generation. */
    private static class ArsDecision {
        final String modelText;
        final String clarifyingQuestion;
        final float confidence;

        ArsDecision(String modelText, String clarifyingQuestion, float confidence) {
            this.modelText = modelText;
            this.clarifyingQuestion = clarifyingQuestion;
            this.confidence = confidence;
        }
    }

    private static class SubjectResolution {
        final String subject;
        final float confidence;

        SubjectResolution(String subject, float confidence) {
            this.subject = subject;
            this.confidence = confidence;
        }
    }
}
