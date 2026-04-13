package com.danexchat;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.Editable;
import android.text.TextWatcher;
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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
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
    // Definition-style turns are treated as a topic switch when token overlap drops below
    // this Jaccard threshold, which resets model-side context for cleaner answers.
    private static final float TOPIC_TOKEN_OVERLAP_THRESHOLD = 0.2f;
    private static final int MIN_TOPIC_TOKEN_LENGTH = 3;
    private static final Pattern TOPIC_TOKEN_SPLIT_PATTERN = Pattern.compile("[^\\p{L}\\p{N}]+");
    private static final Pattern DEFINITION_ARTICLE_PATTERN = Pattern.compile("^(a|an|the)\\s+");
    private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");
    private static final int RECENT_CONTEXT_TOKEN_BUDGET = 1500;
    private static final int SUMMARY_TOKEN_BUDGET = 300;
    private static final int ARCHIVE_TOKEN_BUDGET = 220;
    private static final int ARCHIVE_CONDENSED_BUDGET = ARCHIVE_TOKEN_BUDGET / 2;
    private static final int SUMMARY_BATCH_SIZE = 8;
    private static final int KEEP_RECENT_MESSAGES = 6;
    private static final int MESSAGE_OVERHEAD_TOKENS = 4;
    private static final int SUMMARY_SNIPPET_MAX_CHARS = 140;
    // Reveal pacing tuned for natural readability while keeping response display responsive.
    private static final long RESPONSE_REVEAL_DELAY_MS = 16L;
    private static final long RESPONSE_SENTENCE_PAUSE_MS = 220L;
    private static final long RESPONSE_CLAUSE_PAUSE_MS = 80L;
    private static final Set<String> AMBIGUOUS_REFERENCES = new HashSet<>(Arrays.asList(
            "it", "this", "that", "they", "them", "he", "she", "him", "her", "these", "those"
    ));
    private static final Set<String> TOPIC_STOPWORDS = new HashSet<>(Arrays.asList(
            "a", "an", "the", "is", "are", "was", "were", "am", "be", "to", "of", "for", "in",
            "on", "at", "and", "or", "but", "with", "about", "this", "that", "it", "its", "as",
            "what", "who", "when", "where", "why", "how", "tell", "me", "please",
            "do", "does", "can", "will", "would"
    ));

    private RecyclerView recyclerView;
    private ChatAdapter chatAdapter;
    private LinearLayoutManager layoutManager;
    private EditText inputField;
    private Button sendButton;

    private TextView tvStatus;
    private View inputRow;
    private final List<Message> messages = new ArrayList<>();
    private final List<Message> conversationHistory = new ArrayList<>();
    private String conversationSummary = "";
    private String archivedSummary = "";
    private ModelManager modelManager;
    private SmolLMInference smolLM;
    private BertTinyEncoder bertEncoder;
    // keyword-only fallback until BERT-tiny finishes loading
    private volatile IntentRouter intentRouter = new IntentRouter(null);
    private final ExecutorService bgExecutor = Executors.newSingleThreadExecutor();
    private final Handler uiHandler = new Handler(Looper.getMainLooper());

    private volatile boolean isGenerating = false;
    // Response reveal state; all accesses are confined to the main/UI thread.
    private int activeGenerationId = 0;
    private Runnable activeRevealRunnable;
    private Message activeRevealMessage;
    private int activeRevealGenerationId = -1;
    private int activeRevealIndex = 0;
    private String activeRevealFullText = "";
    private final StringBuilder activeRevealBuffer = new StringBuilder();
    private boolean modelReady = false;

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
        cancelActiveReveal();
        bgExecutor.shutdownNow();
        if (smolLM != null) smolLM.close();
        if (bertEncoder != null) bertEncoder.close();
    }

    private void bindViews() {
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        recyclerView = findViewById(R.id.recyclerViewMessages);
        inputField = findViewById(R.id.etUserInput);
        sendButton = findViewById(R.id.btnSend);
        tvStatus = findViewById(R.id.tvStatus);
        inputRow = findViewById(R.id.inputRow);

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

        sendButton.setOnClickListener(v -> onSendOrStopClicked());
        inputField.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                updateSendEnabledForInput();
            }

            @Override
            public void afterTextChanged(Editable s) {}
        });
    }

    private void setupRecyclerView() {
        chatAdapter = new ChatAdapter(messages);
        layoutManager = new LinearLayoutManager(this);
        layoutManager.setStackFromEnd(true);
        recyclerView.setLayoutManager(layoutManager);
        recyclerView.setAdapter(chatAdapter);
    }

    private void checkAndLoadModel() {
        showStatus(getString(R.string.loading_model));
        setInputEnabled(false);
        modelReady = false;
        updateSendEnabledForInput();

        bgExecutor.execute(() -> {
            if (!modelManager.isReady()) {
                runOnUiThread(() -> {
                    showStatus(getString(R.string.bundled_model_missing));
                    addAssistantMessage(getString(R.string.bundled_model_missing_chat));
                    setInputEnabled(false);
                    modelReady = false;
                    updateSendEnabledForInput();
                });
                return;
            }

            try {
                if (!modelManager.hasBertFiles()) {
                    throw new IllegalStateException(
                            "Bundled BERT-tiny assets are missing in app/src/main/assets/bert_tiny/."
                                    + " See README for required files and export steps.");
                }
                BertTinyEncoder encoder = new BertTinyEncoder(
                        modelManager.getBertModelFile(),
                        modelManager.getBertVocabFile());
                final BertTinyEncoder finalEncoder = encoder;
                final IntentRouter router = new IntentRouter(encoder);

                SmolLMInference inference = new SmolLMInference(
                        modelManager.getModelFile(),
                        modelManager.getTokenizerFile(),
                        modelManager.getDictionaryFile(),
                        encoder);
                runOnUiThread(() -> {
                    bertEncoder  = finalEncoder;
                    intentRouter = router;
                    smolLM = inference;
                    smolLM.updateGenerationOptions(SmolLMInference.GenerationOptions.defaults());
                    hideStatus();
                    modelReady = true;
                    setInputEnabled(true);
                    updateSendEnabledForInput();
                    addAssistantMessage(getString(R.string.model_ready));
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    showStatus(getString(R.string.load_error, e.getMessage()));
                    addAssistantMessage(getString(R.string.error_prefix, e.getMessage()));
                    modelReady = false;
                    setInputEnabled(false);
                    updateSendEnabledForInput();
                });
            }
        });
    }

    private void onSendOrStopClicked() {
        if (isGenerating) {
            if (activeRevealRunnable != null) {
                finishRevealEarly();
                return;
            }
            if (smolLM != null) smolLM.requestStop();
            return;
        }
        onSendClicked();
    }

    private void onSendClicked() {
        String text = inputField.getText().toString().trim();
        if (text.isEmpty() || smolLM == null) return;

        boolean didResetContext = shouldResetConversationContext(conversationHistory, text);
        if (didResetContext) {
            Log.d(TAG, "Resetting model conversation context for detected topic switch");
            conversationHistory.clear();
            conversationSummary = "";
            archivedSummary = "";
        }

        String modelText = resolveAmbiguityAndSpecifier(conversationHistory, text);
        inputField.setText("");

        Message userMsg = new Message(Message.ROLE_USER, text);
        addMessage(userMsg);
        conversationHistory.add(userMsg);
        compactConversationHistoryIfNeeded();
        if (!modelText.equals(text)) {
            Log.d(TAG, "Ambiguity resolved: '" + text + "' -> '" + modelText + "'");
        }

        Message aiMsg = new Message(Message.ROLE_ASSISTANT, "");
        addMessage(aiMsg);
        List<Message> history = new ArrayList<>(conversationHistory);
        if (!modelText.equals(text)) {
            history.set(history.size() - 1, new Message(Message.ROLE_USER, modelText));
        }

        isGenerating = true;
        cancelActiveReveal();
        final int generationId = ++activeGenerationId;
        updateSendEnabledForInput();
        final String finalModelText = modelText;
        bgExecutor.execute(() -> {
            if (didResetContext) {
                smolLM.clearResponseCache();
            }
            // Intent routing runs on the background thread so that BERT-tiny
            // inference (when the encoder is available) does not block the UI.
            IntentRouter.RouteResult routeResult = intentRouter.route(finalModelText);
            smolLM.updateGenerationOptions(routeResult.options);
            Map<String, String> promptTags = buildPromptTags(routeResult.route);
            smolLM.generate(history, conversationSummary, archivedSummary, promptTags,
                    new SmolLMInference.StreamCallback() {
                        @Override
                        public void onToken(String piece) {
                            // UI reveal is intentionally deferred until onComplete so output can be
                            // shown with smooth character pacing instead of raw token bursts.
                        }

                        @Override
                        public void onComplete(String fullResponse) {
                            runOnUiThread(() -> {
                                if (generationId != activeGenerationId) return;
                                beginReveal(aiMsg, fullResponse, generationId);
                            });
                        }

                        @Override
                        public void onError(Exception e) {
                            runOnUiThread(() -> {
                                if (generationId != activeGenerationId) return;
                                cancelActiveReveal();
                                aiMsg.setContent(getString(R.string.error_prefix, e.getMessage()));
                                int pos = messages.indexOf(aiMsg);
                                if (pos >= 0) chatAdapter.notifyItemChanged(pos);
                                isGenerating = false;
                                updateSendEnabledForInput();
                            });
                        }
                    });
        });
    }

    private void addMessage(Message msg) {
        messages.add(msg);
        chatAdapter.notifyItemInserted(messages.size() - 1);
    }

    private void addAssistantMessage(String text) {
        addMessage(new Message(Message.ROLE_ASSISTANT, text));
    }

    private void setInputEnabled(boolean enabled) {
        inputField.setEnabled(enabled);
    }

    private void setSendButtonState(boolean enabled, boolean generating) {
        sendButton.setEnabled(enabled);
        if (generating) {
            sendButton.setText(getString(R.string.stop_square));
            sendButton.setContentDescription(getString(R.string.stop));
        } else {
            sendButton.setText(getString(R.string.send_arrow));
            sendButton.setContentDescription(getString(R.string.send));
        }
    }

    private void updateSendEnabledForInput() {
        boolean hasText = !inputField.getText().toString().trim().isEmpty();
        boolean enabled = isGenerating || (modelReady && inputField.isEnabled() && hasText);
        setSendButtonState(enabled, isGenerating);
    }

    private void beginReveal(Message aiMsg, String fullResponse, int generationId) {
        cancelActiveReveal();
        String safeResponse = fullResponse == null ? "" : fullResponse;
        aiMsg.setContent("");
        int pos = messages.indexOf(aiMsg);
        if (pos >= 0) chatAdapter.notifyItemChanged(pos);

        activeRevealMessage = aiMsg;
        activeRevealGenerationId = generationId;
        activeRevealIndex = 0;
        activeRevealFullText = safeResponse;
        activeRevealBuffer.setLength(0);
        activeRevealRunnable = new Runnable() {
            @Override
            public void run() {
                if (generationId != activeGenerationId || activeRevealMessage == null) {
                    cancelActiveReveal();
                    return;
                }
                if (activeRevealIndex >= safeResponse.length()) {
                    finalizeAssistantMessage(activeRevealMessage, generationId);
                    return;
                }
                char ch = safeResponse.charAt(activeRevealIndex++);
                activeRevealBuffer.append(ch);
                activeRevealMessage.setContent(activeRevealBuffer.toString());
                int itemPos = messages.indexOf(activeRevealMessage);
                if (itemPos >= 0) chatAdapter.notifyItemChanged(itemPos);
                uiHandler.postDelayed(this, nextRevealDelayMs(ch));
            }
        };
        uiHandler.post(activeRevealRunnable);
    }

    private long nextRevealDelayMs(char ch) {
        if (ch == '.' || ch == '!' || ch == '?' || ch == '\n') {
            return RESPONSE_SENTENCE_PAUSE_MS;
        }
        if (ch == ',' || ch == ';' || ch == ':') {
            return RESPONSE_CLAUSE_PAUSE_MS;
        }
        return RESPONSE_REVEAL_DELAY_MS;
    }

    private void finishRevealEarly() {
        if (activeRevealRunnable == null || activeRevealMessage == null) return;
        uiHandler.removeCallbacks(activeRevealRunnable);
        activeRevealRunnable = null;
        if (activeRevealGenerationId == activeGenerationId) {
            finalizeAssistantMessage(activeRevealMessage, activeRevealGenerationId);
        } else {
            cancelActiveReveal();
        }
    }

    private void cancelActiveReveal() {
        if (activeRevealRunnable != null) {
            uiHandler.removeCallbacks(activeRevealRunnable);
        }
        activeRevealRunnable = null;
        activeRevealMessage = null;
        activeRevealGenerationId = -1;
        activeRevealIndex = 0;
        activeRevealFullText = "";
        activeRevealBuffer.setLength(0);
    }

    private void finalizeAssistantMessage(Message aiMsg, int generationId) {
        if (generationId != activeGenerationId) {
            cancelActiveReveal();
            return;
        }
        if (!activeRevealFullText.isEmpty() && !activeRevealFullText.equals(aiMsg.getContent())) {
            aiMsg.setContent(activeRevealFullText);
            int pos = messages.indexOf(aiMsg);
            if (pos >= 0) chatAdapter.notifyItemChanged(pos);
        }
        if (conversationHistory.isEmpty()
                || conversationHistory.get(conversationHistory.size() - 1).isUser()) {
            conversationHistory.add(new Message(Message.ROLE_ASSISTANT, aiMsg.getContent()));
            compactConversationHistoryIfNeeded();
        }
        cancelActiveReveal();
        isGenerating = false;
        updateSendEnabledForInput();
    }

    private void showStatus(String text) {
        tvStatus.setText(text);
        tvStatus.setVisibility(View.VISIBLE);
    }

    private void hideStatus() {
        tvStatus.setVisibility(View.GONE);
    }

    private boolean shouldResetConversationContext(List<Message> conversationHistory, String newUserText) {
        Message lastUserMessage = findLastUserMessage(conversationHistory);
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

    private Message findLastUserMessage(List<Message> conversationHistory) {
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

    private String resolveAmbiguityAndSpecifier(List<Message> conversationHistory, String userText) {
        String subject = extractDefinitionSubject(userText);
        if (subject == null) {
            if (!containsAmbiguousReference(userText)) {
                return userText;
            }
            String resolvedSubject = resolveMostRecentConcreteSubject(conversationHistory);
            if (resolvedSubject == null) {
                return userText;
            }
            return rewriteAmbiguousFollowUp(userText, sanitizeResolvedSubject(resolvedSubject));
        }
        if (!AMBIGUOUS_REFERENCES.contains(subject)) {
            return userText;
        }

        String resolvedSubject = resolveMostRecentConcreteSubject(conversationHistory);
        if (resolvedSubject == null) {
            return userText;
        }
        return rewriteDefinitionSubject(userText, sanitizeResolvedSubject(resolvedSubject));
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

    private String resolveMostRecentConcreteSubject(List<Message> conversationHistory) {
        for (int i = conversationHistory.size() - 1; i >= 0; i--) {
            Message message = conversationHistory.get(i);
            if (!message.isUser()) continue;
            String subject = extractDefinitionSubject(message.getContent());
            if (subject == null || AMBIGUOUS_REFERENCES.contains(subject)) continue;
            return subject;
        }
        return null;
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
        return originalText + " (about " + subject + ")";
    }

    private static String sanitizeResolvedSubject(String subject) {
        return subject.replace('\n', ' ')
                .replace('\r', ' ')
                .replace('\t', ' ')
                .trim();
    }

    private Map<String, String> buildPromptTags(String routeHint) {
        Map<String, String> tags = new HashMap<>();
        tags.put("$route", routeHint);
        return tags;
    }

    private void compactConversationHistoryIfNeeded() {
        while (estimateMessageTokenCount(conversationHistory) > RECENT_CONTEXT_TOKEN_BUDGET
                && conversationHistory.size() > KEEP_RECENT_MESSAGES) {
            int maxCompressible = conversationHistory.size() - KEEP_RECENT_MESSAGES;
            int compressCount = Math.min(SUMMARY_BATCH_SIZE, maxCompressible);
            List<Message> toCompress = new ArrayList<>(conversationHistory.subList(0, compressCount));
            conversationHistory.subList(0, compressCount).clear();
            conversationSummary = mergeSummary(conversationSummary, toCompress, SUMMARY_TOKEN_BUDGET);
        }
        if (estimateTextTokens(conversationSummary) > SUMMARY_TOKEN_BUDGET) {
            archivedSummary = mergeHighLevelSummary(archivedSummary, conversationSummary);
            conversationSummary = shrinkToTokenBudget(conversationSummary, SUMMARY_TOKEN_BUDGET);
        }
    }

    private static String mergeSummary(String existingSummary, List<Message> messagesToCompress, int tokenBudget) {
        StringBuilder sb = new StringBuilder();
        if (!existingSummary.isEmpty()) {
            sb.append(existingSummary);
        }
        for (Message msg : messagesToCompress) {
            String normalized = normalizeWhitespace(msg.getContent());
            if (normalized.isEmpty()) continue;
            if (normalized.length() > SUMMARY_SNIPPET_MAX_CHARS) {
                normalized = normalized.substring(0, SUMMARY_SNIPPET_MAX_CHARS).trim() + "...";
            }
            if (sb.length() > 0) sb.append(" | ");
            sb.append(msg.isUser() ? "U: " : "A: ").append(normalized);
        }
        return shrinkToTokenBudget(sb.toString(), tokenBudget);
    }

    private static String mergeHighLevelSummary(String existingArchive, String summaryToArchive) {
        String condensed = shrinkToTokenBudget(summaryToArchive, ARCHIVE_CONDENSED_BUDGET);
        if (condensed.isEmpty()) return existingArchive;
        StringBuilder merged = new StringBuilder();
        if (!existingArchive.isEmpty()) {
            merged.append(existingArchive).append(" || ");
        }
        merged.append(condensed);
        return shrinkToTokenBudget(merged.toString(), ARCHIVE_TOKEN_BUDGET);
    }

    private static int estimateMessageTokenCount(List<Message> history) {
        int count = 0;
        for (Message message : history) {
            count += estimateTextTokens(message.getContent()) + MESSAGE_OVERHEAD_TOKENS;
        }
        return count;
    }

    private static int estimateTextTokens(String text) {
        String normalized = normalizeWhitespace(text);
        if (normalized.isEmpty()) return 0;
        int words = countWords(normalized);
        int chars = normalized.length();
        int charEstimate = Math.max(1, chars / 4);
        return Math.max(words, charEstimate);
    }

    private static String shrinkToTokenBudget(String text, int tokenBudget) {
        String normalized = normalizeWhitespace(text);
        if (normalized.isEmpty()) return "";
        if (estimateTextTokens(normalized) <= tokenBudget) return normalized;
        int wordBudget = Math.max(1, tokenBudget);
        String tail = tailWords(normalized, wordBudget);
        if (tail.equals(normalized)) return tail;
        return ("... " + tail).trim();
    }

    private static String normalizeWhitespace(String text) {
        if (text == null) return "";
        return WHITESPACE_PATTERN.matcher(text).replaceAll(" ").trim();
    }

    private static int countWords(String normalized) {
        if (normalized.isEmpty()) return 0;
        int count = 1;
        for (int i = 0; i < normalized.length(); i++) {
            if (normalized.charAt(i) == ' ') {
                count++;
            }
        }
        return count;
    }

    private static String tailWords(String normalized, int wordCount) {
        int spacesSeen = 0;
        int startIndex = 0;
        for (int i = normalized.length() - 1; i >= 0; i--) {
            if (normalized.charAt(i) == ' ') {
                spacesSeen++;
                if (spacesSeen == wordCount) {
                    startIndex = i + 1;
                    break;
                }
            }
        }
        if (spacesSeen < wordCount) {
            return normalized;
        }
        return normalized.substring(startIndex).trim();
    }

}
