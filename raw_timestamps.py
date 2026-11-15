"""
Extract frame-level drum onset estimates with librosa.
Used to inspect the timestamps before the versioned analysis.
"""

import librosa


def extract_timestamps_from_audio(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)

    # backtracking moves each peak to the preceding onset-envelope minimum
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, units='frames', backtrack=True, wait=8
    )

    timestamps = librosa.frames_to_time(onset_frames, sr=sr)

    print(f"Successfully extracted {len(timestamps)} timestamps.")
    print("\nFormat for python script:")
    print("raw_timestamps = np.array([")
    print("    " + ", ".join([f"{t:.3f}" for t in timestamps]))
    print("])")

    return timestamps


if __name__ == "__main__":
    audio_file = "drum_take_01.wav"
    try:
        extract_timestamps_from_audio(audio_file)
    except FileNotFoundError:
        print(f"File '{audio_file}' not found. Please check the file path.")