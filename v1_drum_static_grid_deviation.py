"""
Drum timing model v1
Static deviation from a session-average timing grid.
Date: November 2026
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt

AUDIO_FILE = "drum_take_01.wav"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
WAIT_FRAMES = 8

FIGURE_FILE = "v1_drum_static_grid_deviation.png"


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

def build_static_grid(onset_times):
    iois = np.diff(onset_times) * 1000.0
    mean_ioi = np.mean(iois)
    session_bpm = 60000.0 / mean_ioi

    # use the average interval as a rough reference grid
    target_times = onset_times[0] + np.arange(len(onset_times)) * mean_ioi / 1000.0
    timing_errors = (onset_times - target_times) * 1000.0

    return iois, mean_ioi, session_bpm, timing_errors

def plot_grid_errors(timing_errors):
    plt.figure(figsize=(10, 4))
    plt.plot(
        timing_errors,
        marker="o",
        linestyle="-",
        color="#1f77b4",
        label="Grid deviation"
    )
    # zero-deviation reference
    plt.axhline(0, color="#d62728", linestyle="--", alpha=0.6)
    plt.title("Drum timing v1: deviation from fitted static grid", fontsize=10)
    plt.xlabel("Strike index", fontsize=9)
    plt.ylabel("Cumulative timing deviation (ms)", fontsize=9)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)
    plt.show()


if __name__ == "__main__":
    print("Drum timing v1: static grid analysis")

    onset_times = extract_onsets(AUDIO_FILE)
    if len(onset_times) < 3:
        raise ValueError("At least three detected strikes are required.")

    iois, mean_ioi, session_bpm, timing_errors = build_static_grid(onset_times)

    print("\n--- ONSET SUMMARY ---")
    print(f"  detected strikes: {len(onset_times)}")
    print(f"  detected IOIs: {len(iois)}")
    print(f"  mean IOI: {mean_ioi:.2f} ms")
    print(f"  session-average tempo: {session_bpm:.2f} BPM")

    print("\n--- STATIC GRID DEVIATION ---")
    print(f"  mean deviation: {np.mean(timing_errors):.2f} ms")
    print(f"  deviation SD: {np.std(timing_errors, ddof=1):.2f} ms")
    print(f"  minimum deviation: {np.min(timing_errors):.2f} ms")
    print(f"  maximum deviation: {np.max(timing_errors):.2f} ms")

    print("\n[NOTE] The reference grid is estimated from this same recording.")
    print("[NOTE] It is a descriptive baseline, not an external metronome grid.")

    plot_grid_errors(timing_errors)