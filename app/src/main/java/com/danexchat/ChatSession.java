package com.danexchat;

import java.util.ArrayList;
import java.util.List;

class ChatSession {
    final List<Message> messages = new ArrayList<>();
    final List<Message> conversationHistory = new ArrayList<>();
}
