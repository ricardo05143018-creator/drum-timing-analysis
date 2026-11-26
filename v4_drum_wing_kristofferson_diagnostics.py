"""
Drum timing model v4
Wing-Kristofferson covariance decomposition.
Raw and linearly detrended IOI analysis.
Date: November 2026
"""

import numpy as np
import librosa
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt

AUDIO_FILE = "drum_take_01.wav"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
WAIT_FRAMES = 8

MAX_LAG = 6

FIGURE_FILE = "v4_drum_wing_kristofferson_diagnostics.png"


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

def sample_autocovariance(series, lag):
    centered = series - np.mean(series)
    return np.sum(centered[:-lag] * centered[lag:]) / len(series)

def wk_components(iois):
    gamma_0 = np.var(iois, ddof=0)
    gamma_1 = sample_autocovariance(iois, 1)
    motor_component = -gamma_1
    clock_component = gamma_0 + 2.0 * gamma_1
    return gamma_0, gamma_1, motor_component, clock_component

def lag_diagnostics(iois):
    # check nearby lags because the model puts most of the dependence at lag 1
    diagnostics = []
    for lag in range(1, min(MAX_LAG, len(iois) - 2) + 1):
        lag_r, lag_p = stats.pearsonr(iois[:-lag], iois[lag:])
        gamma = sample_autocovariance(iois, lag)
        diagnostics.append((lag, lag_r, lag_p, gamma))
    return diagnostics

def detrend_iois(iois):
    strike_index = np.arange(len(iois))
    trend = stats.linregress(strike_index, iois)

    # remove the fitted linear trend while keeping the mean IOI
    detrended = signal.detrend(iois, type="linear") + np.mean(iois)
    return detrended, trend

def plot_wk_diagnostics(raw_iois, detrended_iois):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, series, color, label in [
        (axes[0], raw_iois, "#d62728", "Raw IOIs"),
        (axes[1], detrended_iois, "#9467bd", "Linearly detrended IOIs")
    ]:
        previous_iois = series[:-1]
        current_iois = series[1:]
        lag_r, lag_p = stats.pearsonr(previous_iois, current_iois)
        slope, intercept = np.polyfit(previous_iois, current_iois, 1)
        order = np.argsort(previous_iois)

        ax.scatter(
            previous_iois,
            current_iois,
            color=color,
            edgecolor="k",
            s=55,
            alpha=0.75,
            label=label
        )
        ax.plot(
            previous_iois[order],
            slope * previous_iois[order] + intercept,
            color="k",
            alpha=0.7,
            label=f"Fit: r={lag_r:.2f}, p={lag_p:.3f}"
        )
        ax.set_xlabel("Previous IOI (ms)", fontsize=9)
        ax.set_ylabel("Current IOI (ms)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[0].set_title("Raw IOI lag-1 transitions", fontsize=10)
    axes[1].set_title("Detrended IOI lag-1 transitions", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)
    plt.show()


if __name__ == "__main__":
    print("Drum timing v4: Wing-Kristofferson diagnostics")

    onset_times = extract_onsets(AUDIO_FILE)
    if len(onset_times) < 10:
        raise ValueError("At least ten detected strikes are required.")

    # WK works with IOIs rather than the cumulative grid-error path
    raw_iois = np.diff(onset_times) * 1000.0
    detrended_iois, trend = detrend_iois(raw_iois)

    raw_gamma_0, raw_gamma_1, raw_motor, raw_clock = wk_components(raw_iois)
    det_gamma_0, det_gamma_1, det_motor, det_clock = wk_components(detrended_iois)

    raw_r, raw_p = stats.pearsonr(raw_iois[:-1], raw_iois[1:])
    det_r, det_p = stats.pearsonr(detrended_iois[:-1], detrended_iois[1:])

    print("\n--- LINEAR TREND DIAGNOSTIC ---")
    print(f"  slope: {trend.slope:.4f} ms per strike")
    print(f"  p-value: {trend.pvalue:.4f}")
    print(f"  fitted end-to-end change: {trend.slope * (len(raw_iois) - 1):.2f} ms")

    print("\n--- RAW IOI WK COMPONENTS ---")
    print(f"  lag-0 variance: {raw_gamma_0:.2f} ms^2")
    print(f"  lag-1 covariance: {raw_gamma_1:.2f} ms^2")
    print(f"  lag-1 r: {raw_r:.4f} (p={raw_p:.4f})")
    print(f"  model-implied motor component: {raw_motor:.2f} ms^2")
    print(f"  model-implied clock component: {raw_clock:.2f} ms^2")

    print("\n--- DETRENDED IOI WK COMPONENTS ---")
    print(f"  lag-0 variance: {det_gamma_0:.2f} ms^2")
    print(f"  lag-1 covariance: {det_gamma_1:.2f} ms^2")
    print(f"  lag-1 r: {det_r:.4f} (p={det_p:.4f})")
    print(f"  model-implied motor component: {det_motor:.2f} ms^2")
    print(f"  model-implied clock component: {det_clock:.2f} ms^2")

    print("\n--- RAW IOI LAG DIAGNOSTICS ---")
    print("  Lag | Correlation | P-value | Autocovariance")
    for lag, lag_r, lag_p, gamma in lag_diagnostics(raw_iois):
        print(f"   {lag:2d} |    {lag_r: .4f}   | {lag_p:.4f}  | {gamma: .2f}")

    plot_wk_diagnostics(raw_iois, detrended_iois)