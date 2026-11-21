"""
Drum timing model v3
Tests whether v2 persistence is induced by cumulative construction.
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

SIMULATION_REPLICATIONS = 20000
RANDOM_SEED = 2026

FIGURE_FILE = "v3_drum_ioi_construction_diagnostics.png"


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
    return iois, timing_errors

def row_correlations(previous_values, current_values):
    previous_centered = previous_values - np.mean(previous_values, axis=1, keepdims=True)
    current_centered = current_values - np.mean(current_values, axis=1, keepdims=True)
    numerator = np.sum(previous_centered * current_centered, axis=1)
    denominator = np.sqrt(
        np.sum(previous_centered ** 2, axis=1) *
        np.sum(current_centered ** 2, axis=1)
    )
    return numerator / denominator

def simulate_independent_grid_correlations(iois):
    rng = np.random.default_rng(RANDOM_SEED)
    # simulate independent intervals with the same rough scale as this take
    simulated_iois = rng.normal(
        loc=np.mean(iois),
        scale=np.std(iois, ddof=1),
        size=(SIMULATION_REPLICATIONS, len(iois))
    )

    # each simulated grid is fitted from its own sample mean, as in v1 and v2
    centered_iois = simulated_iois - np.mean(simulated_iois, axis=1, keepdims=True)
    cumulative_errors = np.concatenate(
        [
            np.zeros((SIMULATION_REPLICATIONS, 1)),
            np.cumsum(centered_iois, axis=1)
        ],
        axis=1
    )

    return row_correlations(cumulative_errors[:, :-1], cumulative_errors[:, 1:])

def analyze_iois(iois):
    previous_iois = iois[:-1]
    current_iois = iois[1:]
    lag1_r, lag1_p = stats.pearsonr(previous_iois, current_iois)
    slope, intercept = np.polyfit(previous_iois, current_iois, 1)

    strike_index = np.arange(len(iois))
    trend = stats.linregress(strike_index, iois)

    return lag1_r, lag1_p, slope, intercept, trend, previous_iois, current_iois

def plot_diagnostics(null_correlations, observed_grid_r, previous_iois, current_iois, slope, intercept):
    order = np.argsort(previous_iois)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        null_correlations,
        bins=45,
        color="#1f77b4",
        alpha=0.8,
        edgecolor="white"
    )
    axes[0].axvline(
        observed_grid_r,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Observed r: {observed_grid_r:.3f}"
    )
    axes[0].set_title("Independent IOIs can create persistent grid errors", fontsize=10)
    axes[0].set_xlabel("Lag-1 correlation of cumulative grid errors", fontsize=9)
    axes[0].set_ylabel("Simulation count", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter(
        previous_iois,
        current_iois,
        color="#ff7f0e",
        edgecolor="k",
        s=55,
        alpha=0.8,
        label="Observed IOI transitions"
    )
    axes[1].plot(
        previous_iois[order],
        slope * previous_iois[order] + intercept,
        color="k",
        alpha=0.7,
        label=f"Linear fit (slope: {slope:.2f})"
    )
    axes[1].set_title("Lag-1 dependency in the IOIs", fontsize=10)
    axes[1].set_xlabel("Previous IOI (ms)", fontsize=9)
    axes[1].set_ylabel("Current IOI (ms)", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)
    plt.show()


if __name__ == "__main__":
    print("Drum timing v3: construction check and IOI analysis")

    onset_times = extract_onsets(AUDIO_FILE)
    if len(onset_times) < 6:
        raise ValueError("At least six detected strikes are required.")

    iois, timing_errors = build_grid_errors(onset_times)
    observed_grid_r, observed_grid_p = stats.pearsonr(timing_errors[:-1], timing_errors[1:])
    null_correlations = simulate_independent_grid_correlations(iois)
    exceedance_rate = np.mean(null_correlations >= observed_grid_r)

    lag1_r, lag1_p, slope, intercept, trend, previous_iois, current_iois = analyze_iois(iois)

    print("\n--- V2 CONSTRUCTION CHECK ---")
    print(f"  observed grid-error r: {observed_grid_r:.4f}")
    print(f"  v2 naive p-value: {observed_grid_p:.4e}")
    print(f"  simulated null median r: {np.median(null_correlations):.4f}")
    print(f"  simulated P(r >= observed): {exceedance_rate:.4f}")
    print(f"  simulation replications: {SIMULATION_REPLICATIONS}")

    print("\n--- IOI DEPENDENCY ---")
    print(f"  mean IOI: {np.mean(iois):.2f} ms")
    print(f"  IOI SD: {np.std(iois, ddof=1):.2f} ms")
    print(f"  lag-1 correlation: {lag1_r:.4f}")
    print(f"  Pearson p-value: {lag1_p:.4f}")

    print("\n--- LINEAR TEMPO TREND ---")
    print(f"  slope: {trend.slope:.4f} ms per strike")
    print(f"  p-value: {trend.pvalue:.4f}")
    print(f"  fitted end-to-end change: {trend.slope * (len(iois) - 1):.2f} ms")


    plot_diagnostics(
        null_correlations,
        observed_grid_r,
        previous_iois,
        current_iois,
        slope,
        intercept
    )