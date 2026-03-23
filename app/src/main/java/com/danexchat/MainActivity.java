package com.danexchat;

import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager2.widget.ViewPager2;

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
    private static final float TOPIC_TOKEN_OVERLAP_THRESHOLD = 0.2f;
    private static final float ARS_RESOLUTION_CONFIDENCE_NONE = 0f;
    private static final float ARS_RESOLUTION_CONFIDENCE_NOT_APPLICABLE = 1f;
    private static final float ARS_RESOLUTION_CONFIDENCE_SINGLE_SUBJECT = 0.95f;
    private static final float ARS_RESOLUTION_CONFIDENCE_MULTI_SUBJECT = 0.55f;
    private static final float ARS_RECENCY_PENALTY_PER_TURN = 0.05f;
    private static final float ARS_MIN_CONFIDENCE = 0.35f;
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

    private RecyclerView recyclerView;
    private ChatAdapter chatAdapter;
    private LinearLayoutManager layoutManager;
    private EditText inputField;
    private Button sendButton;
    private Button scrollToBottomButton;
    private Button newSessionTopButton;
    private ImageButton settingsButton;
    private ViewPager2 sessionsPager;
    private ChatSessionsPagerAdapter sessionsPagerAdapter;

    private TextView tvStatus;
    private View inputRow;

    private final List<ChatSession> sessions = new ArrayList<>();
    private int currentSessionIndex = -1;

    private ModelManager modelManager;
    private SmolLMInference smolLM;
    private final ExecutorService bgExecutor = Executors.newSingleThreadExecutor();

    private volatile boolean isGenerating = false;
    private volatile boolean shouldAutoScrollDuringGeneration = true;
    private boolean modelReady = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        bindViews();
        setupRecyclerView();
        setupSessionsPager();
        createAndSwitchToNewSession();

        modelManager = new ModelManager(this);
        checkAndLoadModel();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        bgExecutor.shutdownNow();
        if (smolLM != null) smolLM.close();
    }

    private void bindViews() {
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        recyclerView = findViewById(R.id.recyclerViewMessages);
        inputField = findViewById(R.id.etUserInput);
        sendButton = findViewById(R.id.btnSend);
        tvStatus = findViewById(R.id.tvStatus);
        inputRow = findViewById(R.id.inputRow);
        scrollToBottomButton = findViewById(R.id.btnScrollToBottom);
        newSessionTopButton = findViewById(R.id.btnNewSessionTop);
        settingsButton = findViewById(R.id.btnSettings);
        sessionsPager = findViewById(R.id.viewPagerSessions);

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
        scrollToBottomButton.setOnClickListener(v -> {
            scrollToBottomImmediate();
            shouldAutoScrollDuringGeneration = true;
            updateScrollToBottomVisibility();
        });
        newSessionTopButton.setOnClickListener(v -> onCreateNewSession());
        settingsButton.setOnClickListener(v ->
                startActivity(new Intent(MainActivity.this, SettingsActivity.class)));
    }

    private void setupRecyclerView() {
        chatAdapter = new ChatAdapter(new ArrayList<>());
        layoutManager = new LinearLayoutManager(this);
        layoutManager.setStackFromEnd(true);
        recyclerView.setLayoutManager(layoutManager);
        recyclerView.setAdapter(chatAdapter);
        recyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override
            public void onScrolled(RecyclerView recyclerView, int dx, int dy) {
                super.onScrolled(recyclerView, dx, dy);
                if (isGenerating && dy < 0) {
                    shouldAutoScrollDuringGeneration = false;
                }
                updateScrollToBottomVisibility();
            }
        });
    }

    private void setupSessionsPager() {
        sessionsPagerAdapter = new ChatSessionsPagerAdapter(sessions);
        sessionsPager.setAdapter(sessionsPagerAdapter);
        sessionsPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageSelected(int position) {
                if (position >= 0 && position < sessions.size()) {
                    switchToSession(position);
                }
            }
        });
    }

    private void checkAndLoadModel() {
        if (modelManager.isReady()) {
            loadModelAsync();
        } else {
            showStatus(getString(R.string.bundled_model_missing));
            addAssistantMessage(getString(R.string.bundled_model_missing_chat));
            setInputEnabled(false);
            modelReady = false;
            updateSendEnabledForInput();
        }
    }

    private void loadModelAsync() {
        showStatus(getString(R.string.loading_model));
        setInputEnabled(false);
        modelReady = false;
        updateSendEnabledForInput();

        bgExecutor.execute(() -> {
            try {
                SmolLMInference inference = new SmolLMInference(
                        modelManager.getModelFile(),
                        modelManager.getTokenizerFile());
                runOnUiThread(() -> {
                    smolLM = inference;
                    hideStatus();
                    modelReady = true;
                    setInputEnabled(true);
                    updateSendEnabledForInput();
                    addAssistantMessage(getString(R.string.model_ready));
                    refreshSessionControls();
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
            if (smolLM != null) smolLM.requestStop();
            return;
        }
        onSendClicked();
    }

    private void onSendClicked() {
        ChatSession session = getCurrentSession();
        if (session == null) return;

        String text = inputField.getText().toString().trim();
        if (text.isEmpty() || smolLM == null) return;

        if (shouldResetConversationContext(session.conversationHistory, text)) {
            Log.d(TAG, "Resetting model conversation context for detected topic switch");
            session.conversationHistory.clear();
        }

        ArsDecision arsDecision = resolveAmbiguityAndSpecifier(session.conversationHistory, text);
        inputField.setText("");

        Message userMsg = new Message(Message.ROLE_USER, text);
        addMessage(userMsg);
        session.conversationHistory.add(userMsg);
        if (!arsDecision.modelText.equals(text)) {
            Log.d(TAG, "ARS specifier applied with confidence: " + arsDecision.confidence);
        }

        Message aiMsg = new Message(Message.ROLE_ASSISTANT, "");
        addMessage(aiMsg);
        List<Message> history = new ArrayList<>(session.conversationHistory);
        if (!arsDecision.modelText.equals(text)) {
            history.set(history.size() - 1, new Message(Message.ROLE_USER, arsDecision.modelText));
        }

        isGenerating = true;
        shouldAutoScrollDuringGeneration = true;
        updateSendEnabledForInput();
        refreshSessionControls();

        bgExecutor.execute(() ->
                smolLM.generate(history, new SmolLMInference.StreamCallback() {
                    @Override
                    public void onToken(String piece) {
                        runOnUiThread(() -> {
                            aiMsg.setContent(aiMsg.getContent() + piece);
                            int pos = getCurrentMessages().indexOf(aiMsg);
                            if (pos >= 0) chatAdapter.notifyItemChanged(pos);
                            if (isGenerating && shouldAutoScrollDuringGeneration) {
                                scrollToBottomImmediate();
                            }
                        });
                    }

                    @Override
                    public void onComplete(String fullResponse) {
                        runOnUiThread(() -> {
                            if (!fullResponse.equals(aiMsg.getContent())) {
                                aiMsg.setContent(fullResponse);
                                int pos = getCurrentMessages().indexOf(aiMsg);
                                if (pos >= 0) chatAdapter.notifyItemChanged(pos);
                            }
                            ChatSession current = getCurrentSession();
                            if (current != null && (current.conversationHistory.isEmpty()
                                    || current.conversationHistory.get(current.conversationHistory.size() - 1).isUser())) {
                                current.conversationHistory.add(new Message(Message.ROLE_ASSISTANT, aiMsg.getContent()));
                            }
                            isGenerating = false;
                            updateSendEnabledForInput();
                            updateScrollToBottomVisibility();
                            refreshSessionControls();
                        });
                    }

                    @Override
                    public void onError(Exception e) {
                        runOnUiThread(() -> {
                            aiMsg.setContent(getString(R.string.error_prefix, e.getMessage()));
                            int pos = getCurrentMessages().indexOf(aiMsg);
                            if (pos >= 0) chatAdapter.notifyItemChanged(pos);
                            isGenerating = false;
                            updateSendEnabledForInput();
                            updateScrollToBottomVisibility();
                            refreshSessionControls();
                        });
                    }
                })
        );
    }

    private void addMessage(Message msg) {
        List<Message> currentMessages = getCurrentMessages();
        currentMessages.add(msg);
        chatAdapter.notifyItemInserted(currentMessages.size() - 1);
        if (isGenerating && shouldAutoScrollDuringGeneration) {
            scrollToBottomImmediate();
        } else {
            updateScrollToBottomVisibility();
        }
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

    private void scrollToBottomImmediate() {
        List<Message> currentMessages = getCurrentMessages();
        if (!currentMessages.isEmpty()) {
            recyclerView.scrollToPosition(currentMessages.size() - 1);
        }
        updateScrollToBottomVisibility();
    }

    private boolean isRecyclerAtBottom() {
        if (chatAdapter.getItemCount() == 0) return true;
        int lastVisible = layoutManager.findLastCompletelyVisibleItemPosition();
        return lastVisible >= chatAdapter.getItemCount() - 1;
    }

    private void updateScrollToBottomVisibility() {
        if (isGenerating) {
            scrollToBottomButton.setVisibility(View.GONE);
            return;
        }
        scrollToBottomButton.setVisibility(isRecyclerAtBottom() ? View.GONE : View.VISIBLE);
    }

    private void showStatus(String text) {
        tvStatus.setText(text);
        tvStatus.setVisibility(View.VISIBLE);
    }

    private void hideStatus() {
        tvStatus.setVisibility(View.GONE);
    }

    private void onCreateNewSession() {
        if (isGenerating) return;
        createAndSwitchToNewSession();
        inputField.requestFocus();
    }

    private void createAndSwitchToNewSession() {
        sessions.add(new ChatSession());
        sessionsPagerAdapter.notifyItemInserted(sessions.size() - 1);
        switchToSession(sessions.size() - 1);
        sessionsPager.setCurrentItem(sessions.size() - 1, true);
    }

    private void switchToSession(int index) {
        currentSessionIndex = index;
        chatAdapter.setMessages(getCurrentMessages());
        scrollToBottomImmediate();
        refreshSessionControls();
    }

    private void refreshSessionControls() {
        ChatSession session = getCurrentSession();
        boolean hasSessionContent = session != null
                && (!session.messages.isEmpty() || !session.conversationHistory.isEmpty());
        newSessionTopButton.setVisibility(hasSessionContent ? View.VISIBLE : View.GONE);
        sessionsPager.setVisibility(sessions.size() > 1 ? View.VISIBLE : View.GONE);
        sessionsPager.setUserInputEnabled(!isGenerating);
        updateScrollToBottomVisibility();
    }

    private ChatSession getCurrentSession() {
        if (currentSessionIndex < 0 || currentSessionIndex >= sessions.size()) return null;
        return sessions.get(currentSessionIndex);
    }

    private List<Message> getCurrentMessages() {
        ChatSession session = getCurrentSession();
        if (session == null) return new ArrayList<>();
        return session.messages;
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

    private ArsDecision resolveAmbiguityAndSpecifier(List<Message> conversationHistory, String userText) {
        String subject = extractDefinitionSubject(userText);
        if (subject == null) {
            if (!containsAmbiguousReference(userText)) {
                return new ArsDecision(userText, ARS_RESOLUTION_CONFIDENCE_NOT_APPLICABLE);
            }
            SubjectResolution resolution = resolveMostRecentConcreteSubject(conversationHistory);
            if (resolution == null) {
                return new ArsDecision(userText, ARS_RESOLUTION_CONFIDENCE_NONE);
            }
            String rewritten = rewriteAmbiguousFollowUp(userText, sanitizeResolvedSubject(resolution.subject));
            return new ArsDecision(rewritten, resolution.confidence);
        }
        if (!AMBIGUOUS_REFERENCES.contains(subject)) {
            return new ArsDecision(userText, ARS_RESOLUTION_CONFIDENCE_NOT_APPLICABLE);
        }

        SubjectResolution resolution = resolveMostRecentConcreteSubject(conversationHistory);
        if (resolution == null) {
            return new ArsDecision(userText, ARS_RESOLUTION_CONFIDENCE_NONE);
        }
        return new ArsDecision(
                rewriteDefinitionSubject(userText, sanitizeResolvedSubject(resolution.subject)),
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

    private SubjectResolution resolveMostRecentConcreteSubject(List<Message> conversationHistory) {
        String mostRecentSubject = null;
        Set<String> uniqueSubjects = new HashSet<>();
        int concreteSubjectsSeen = 0;
        int turnsSinceMostRecentSubject = 0;
        for (int i = conversationHistory.size() - 1; i >= 0; i--) {
            Message message = conversationHistory.get(i);
            if (!message.isUser()) continue;
            String subject = extractDefinitionSubject(message.getContent());
            if (subject == null || AMBIGUOUS_REFERENCES.contains(subject)) continue;
            concreteSubjectsSeen++;
            if (mostRecentSubject == null) {
                mostRecentSubject = subject;
                turnsSinceMostRecentSubject = concreteSubjectsSeen - 1;
            }
            uniqueSubjects.add(subject);
        }
        if (mostRecentSubject == null) return null;
        float baseConfidence = uniqueSubjects.size() == 1
                ? ARS_RESOLUTION_CONFIDENCE_SINGLE_SUBJECT
                : ARS_RESOLUTION_CONFIDENCE_MULTI_SUBJECT;
        float recencyPenalty = Math.min(0.3f, turnsSinceMostRecentSubject * ARS_RECENCY_PENALTY_PER_TURN);
        float confidence = Math.max(ARS_MIN_CONFIDENCE, baseConfidence - recencyPenalty);
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
        final float confidence;

        ArsDecision(String modelText, float confidence) {
            this.modelText = modelText;
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
