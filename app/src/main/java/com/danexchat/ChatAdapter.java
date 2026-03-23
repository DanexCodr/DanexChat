package com.danexchat;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

/**
 * RecyclerView adapter that displays a list of {@link Message} objects
 * as a chat conversation.  User messages appear on the right (VIEW_TYPE_USER)
 * and assistant messages appear on the left (VIEW_TYPE_ASSISTANT).
 */
public class ChatAdapter extends RecyclerView.Adapter<ChatAdapter.MessageViewHolder> {

    private static final int VIEW_TYPE_USER      = 0;
    private static final int VIEW_TYPE_ASSISTANT = 1;

    private List<Message> messages;

    public ChatAdapter(List<Message> messages) {
        this.messages = messages;
    }

    public void setMessages(List<Message> messages) {
        List<Message> oldMessages = this.messages;
        DiffUtil.DiffResult diffResult = DiffUtil.calculateDiff(new DiffUtil.Callback() {
            @Override
            public int getOldListSize() {
                return oldMessages.size();
            }

            @Override
            public int getNewListSize() {
                return messages.size();
            }

            @Override
            public boolean areItemsTheSame(int oldItemPosition, int newItemPosition) {
                return oldMessages.get(oldItemPosition) == messages.get(newItemPosition);
            }

            @Override
            public boolean areContentsTheSame(int oldItemPosition, int newItemPosition) {
                Message oldMessage = oldMessages.get(oldItemPosition);
                Message newMessage = messages.get(newItemPosition);
                return oldMessage.getRole() == newMessage.getRole()
                        && oldMessage.getContent().equals(newMessage.getContent());
            }
        });
        this.messages = messages;
        diffResult.dispatchUpdatesTo(this);
    }

    @Override
    public int getItemViewType(int position) {
        return messages.get(position).isUser() ? VIEW_TYPE_USER : VIEW_TYPE_ASSISTANT;
    }

    @NonNull
    @Override
    public MessageViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        int layoutId = (viewType == VIEW_TYPE_USER)
                ? R.layout.item_message_user
                : R.layout.item_message_assistant;
        View view = LayoutInflater.from(parent.getContext()).inflate(layoutId, parent, false);
        return new MessageViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull MessageViewHolder holder, int position) {
        holder.bind(messages.get(position));
    }

    @Override
    public int getItemCount() {
        return messages.size();
    }

    static class MessageViewHolder extends RecyclerView.ViewHolder {
        private final TextView textView;

        MessageViewHolder(@NonNull View itemView) {
            super(itemView);
            textView = itemView.findViewById(R.id.tvMessageContent);
        }

        void bind(Message message) {
            textView.setText(message.getContent());
        }
    }
}
