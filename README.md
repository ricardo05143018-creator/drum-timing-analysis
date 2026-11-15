# Drum Timing Analysis — V1

I started this project because I was curious about how consistent my timing actually was when playing the drums. Even when a rhythm sounds steady, I wondered how much the intervals between strikes might vary.

Instead of relying only on what I could hear, I decided to record a short drum sequence and look at the timing data myself.

## Method

I recorded a roughly 22-second drum sequence and used Python's `librosa` library to detect the timestamps of individual strikes.

From these timestamps, I calculated the inter-onset intervals (IOIs) and their average duration. I then constructed a fixed timing grid using the first detected strike and the average IOI.

By comparing each strike with its corresponding position on this grid, I could track how the timing deviations accumulated throughout the recording.

## Results

The program detected 70 strikes, giving 69 inter-onset intervals with an average duration of 319.36 ms.

The cumulative timing deviations ranged from approximately −373 ms to +41 ms.

![Cumulative timing deviation](figures/v1_drum_static_grid_deviation.png)

What interested me most was how the deviations changed over the course of the recording. Rather than staying close to zero, the curve moved substantially into negative values before returning toward the reference grid.

The grid was based on the recording's own average interval, so it describes changes relative to that average pace rather than accuracy against an external metronome.

## Next Step

After seeing the curve, I began wondering whether the deviations were related from one strike to the next.

If one strike was ahead of the reference grid, would the following strike tend to remain ahead as well?

That became the starting point for V2.

## Running the Code

Install the required Python packages:

```bash
pip install numpy librosa matplotlib
```

Place `drum_take_01.wav` in the same working directory as `v1_drum_static_grid_deviation.py`, then run:

```bash
python v1_drum_static_grid_deviation.py
```

The script prints the timing statistics and saves the plot as `v1_drum_static_grid_deviation.png`.