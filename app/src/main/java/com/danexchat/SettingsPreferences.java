package com.danexchat;

import android.content.Context;
import android.content.SharedPreferences;

final class SettingsPreferences {

    private static final String PREFS_NAME = "danexchat_settings";

    private static final String KEY_TEMPERATURE = "temperature";
    private static final String KEY_TOP_P = "top_p";
    private static final String KEY_MAX_NEW_TOKENS = "max_new_tokens";

    static final float DEFAULT_TEMPERATURE = 0.8f;
    static final float DEFAULT_TOP_P = 0.9f;
    static final int DEFAULT_MAX_NEW_TOKENS = 256;

    private SettingsPreferences() {
    }

    static SharedPreferences get(Context context) {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    static float getTemperature(Context context) {
        return get(context).getFloat(KEY_TEMPERATURE, DEFAULT_TEMPERATURE);
    }

    static void setTemperature(Context context, float temperature) {
        get(context).edit().putFloat(KEY_TEMPERATURE, temperature).apply();
    }

    static float getTopP(Context context) {
        return get(context).getFloat(KEY_TOP_P, DEFAULT_TOP_P);
    }

    static void setTopP(Context context, float topP) {
        get(context).edit().putFloat(KEY_TOP_P, topP).apply();
    }

    static int getMaxNewTokens(Context context) {
        return get(context).getInt(KEY_MAX_NEW_TOKENS, DEFAULT_MAX_NEW_TOKENS);
    }

    static void setMaxNewTokens(Context context, int maxNewTokens) {
        get(context).edit().putInt(KEY_MAX_NEW_TOKENS, maxNewTokens).apply();
    }
}
