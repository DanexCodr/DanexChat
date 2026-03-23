package com.danexchat;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

public class GeneralSettingsFragment extends Fragment {
    private static final float TENTHS_MULTIPLIER = 10f;
    private static final int TEMPERATURE_MIN_TENTHS = 1;
    private static final int TOP_P_MIN_TENTHS = 5;
    private static final int MAX_TOKENS_MIN = 32;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        View root = inflater.inflate(R.layout.fragment_settings_general, container, false);

        SeekBar temperatureSeek = root.findViewById(R.id.seekTemperature);
        SeekBar topPSeek = root.findViewById(R.id.seekTopP);
        SeekBar maxTokensSeek = root.findViewById(R.id.seekMaxTokens);
        TextView temperatureLabel = root.findViewById(R.id.tvTemperatureLabel);
        TextView topPLabel = root.findViewById(R.id.tvTopPLabel);
        TextView maxTokensLabel = root.findViewById(R.id.tvMaxTokensLabel);

        float currentTemperature = SettingsPreferences.getTemperature(requireContext());
        float currentTopP = SettingsPreferences.getTopP(requireContext());
        int currentMaxTokens = SettingsPreferences.getMaxNewTokens(requireContext());

        int temperatureProgress = Math.round(currentTemperature * TENTHS_MULTIPLIER) - TEMPERATURE_MIN_TENTHS;
        int topPProgress = Math.round(currentTopP * TENTHS_MULTIPLIER) - TOP_P_MIN_TENTHS;
        int maxTokensProgress = currentMaxTokens - MAX_TOKENS_MIN;

        temperatureSeek.setProgress(Math.max(0, temperatureProgress));
        topPSeek.setProgress(Math.max(0, topPProgress));
        maxTokensSeek.setProgress(Math.max(0, maxTokensProgress));

        updateTemperatureLabel(temperatureLabel, currentTemperature);
        updateTopPLabel(topPLabel, currentTopP);
        updateMaxTokensLabel(maxTokensLabel, currentMaxTokens);

        temperatureSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float value = (TEMPERATURE_MIN_TENTHS + progress) / TENTHS_MULTIPLIER;
                SettingsPreferences.setTemperature(requireContext(), value);
                updateTemperatureLabel(temperatureLabel, value);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        topPSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float value = (TOP_P_MIN_TENTHS + progress) / TENTHS_MULTIPLIER;
                SettingsPreferences.setTopP(requireContext(), value);
                updateTopPLabel(topPLabel, value);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        maxTokensSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int value = MAX_TOKENS_MIN + progress;
                SettingsPreferences.setMaxNewTokens(requireContext(), value);
                updateMaxTokensLabel(maxTokensLabel, value);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        return root;
    }

    private void updateTemperatureLabel(TextView label, float value) {
        label.setText(getString(R.string.settings_temperature_value, value));
    }

    private void updateTopPLabel(TextView label, float value) {
        label.setText(getString(R.string.settings_top_p_value, value));
    }

    private void updateMaxTokensLabel(TextView label, int value) {
        label.setText(getString(R.string.settings_max_tokens_value, value));
    }
}
