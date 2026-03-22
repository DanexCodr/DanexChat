package com.danexchat;

/**
 * Represents a single chat message.
 */
public class Message {
    public static final int ROLE_USER = 0;
    public static final int ROLE_ASSISTANT = 1;

    private final int role;
    private String content;

    public Message(int role, String content) {
        this.role = role;
        this.content = content;
    }

    public int getRole() {
        return role;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public boolean isUser() {
        return role == ROLE_USER;
    }
}
