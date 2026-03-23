package com.danexchat;

import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

class ChatSessionsPagerAdapter extends RecyclerView.Adapter<ChatSessionsPagerAdapter.PageViewHolder> {

    private final List<ChatSession> sessions;

    ChatSessionsPagerAdapter(List<ChatSession> sessions) {
        this.sessions = sessions;
    }

    @NonNull
    @Override
    public PageViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        TextView page = new TextView(parent.getContext());
        page.setLayoutParams(new ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));
        page.setTextAlignment(View.TEXT_ALIGNMENT_CENTER);
        page.setTextSize(12f);
        page.setGravity(android.view.Gravity.CENTER);
        page.setTextColor(ContextCompat.getColor(parent.getContext(), R.color.textSecondary));
        return new PageViewHolder(page);
    }

    @Override
    public void onBindViewHolder(@NonNull PageViewHolder holder, int position) {
        holder.label.setText(holder.itemView.getContext().getString(R.string.session_label, position + 1));
    }

    @Override
    public int getItemCount() {
        return sessions.size();
    }

    static class PageViewHolder extends RecyclerView.ViewHolder {
        final TextView label;

        PageViewHolder(@NonNull View itemView) {
            super(itemView);
            label = (TextView) itemView;
        }
    }
}
