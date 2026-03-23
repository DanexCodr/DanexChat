package com.danexchat;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Switch;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

public class SessionSettingsFragment extends Fragment {
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        View root = inflater.inflate(R.layout.fragment_settings_session, container, false);

        Switch swipeSwitch = root.findViewById(R.id.switchSessionSwipe);
        Switch autoScrollSwitch = root.findViewById(R.id.switchAutoScroll);

        swipeSwitch.setChecked(SettingsPreferences.isSessionSwipeEnabled(requireContext()));
        autoScrollSwitch.setChecked(SettingsPreferences.isAutoScrollEnabled(requireContext()));

        swipeSwitch.setOnCheckedChangeListener((buttonView, isChecked) ->
                SettingsPreferences.setSessionSwipeEnabled(requireContext(), isChecked));
        autoScrollSwitch.setOnCheckedChangeListener((buttonView, isChecked) ->
                SettingsPreferences.setAutoScrollEnabled(requireContext(), isChecked));

        return root;
    }
}
