package com.danexchat;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.viewpager2.adapter.FragmentStateAdapter;

class SettingsPagerAdapter extends FragmentStateAdapter {

    SettingsPagerAdapter(@NonNull AppCompatActivity activity) {
        super(activity);
    }

    @NonNull
    @Override
    public Fragment createFragment(int position) {
        if (position == 0) {
            return new GeneralSettingsFragment();
        }
        return new SessionSettingsFragment();
    }

    @Override
    public int getItemCount() {
        return 2;
    }
}
