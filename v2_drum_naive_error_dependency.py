"""
Drum timing model v2
Added lag-1 dependency diagnostics on the v1 grid errors.
Date: November 2026
"""

import numpy as np
import librosa
import scipy.stats as stats
import matplotlib.pyplot as plt

AUDIO_FILE = "drum_take_01.wav"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
WAIT_FRAMES = 8

FIGURE_FILE = "v2_drum_naive_error_dependency.png"


def extract_onsets(audio_file):
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="frames",
        backtrack=True,
        wait=WAIT_FRAMES
    )
    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=sr,
        hop_length=HOP_LENGTH
    )
    return onset_times

def build_grid_errors(onset_times):
    iois = np.diff(onset_times) * 1000.0
    mean_ioi = np.mean(iois)
    target_times = onset_times[0] + np.arange(len(onset_times)) * mean_ioi / 1000.0
    timing_errors = (onset_times - target_times) * 1000.0
    return timing_errors

def lag1_dependency(series):
    previous_values = series[:-1]
    current_values = series[1:]
    lag1_r, p_value = stats.pearsonr(previous_values, current_values)
    slope, intercept = np.polyfit(previous_values, current_values, 1)
    return lag1_r, p_value, slope, intercept, previous_values, current_values

def plot_dependency(previous_values, current_values, slope, intercept):
    # sort the fitted line so the visual does not jump across the scatter
    order = np.argsort(previous_values)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        previous_values,
        current_values,
        color="#2ca02c",
        edgecolor="k",
        s=60,
        alpha=0.8,
        label="Observed transitions"
    )
    plt.plot(
        previous_values[order],
        slope * previous_values[order] + intercept,
        color="#d62728",
        alpha=0.7,
        label=f"Linear fit (slope: {slope:.2f})"
    )
    plt.axhline(0, color="gray", linestyle="--", alpha=0.3)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.3)
    plt.title("Drum timing v2: lag-1 grid-error dependency", fontsize=10)
    plt.xlabel("Previous cumulative deviation (ms)", fontsize=9)
    plt.ylabel("Current cumulative deviation (ms)", fontsize=9)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)
    plt.show()


if __name__ == "__main__":
    print("Drum timing v2: lag-1 grid-error analysis")

    onset_times = extract_onsets(AUDIO_FILE)
    if len(onset_times) < 4:
        raise ValueError("At least four detected strikes are required.")

    timing_errors = build_grid_errors(onset_times)
    lag1_r, p_value, slope, intercept, previous_values, current_values = lag1_dependency(timing_errors)

    print("\n--- GRID-ERROR DEPENDENCY ---")
    print(f"  detected strikes: {len(onset_times)}")
    print(f"  lag-1 correlation: {lag1_r:.4f}")
    print(f"  naive Pearson p-value: {p_value:.4e}")
    print(f"  fitted slope: {slope:.4f}")


    plot_dependency(previous_values, current_values, slope, intercept)