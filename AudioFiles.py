import numpy as np
import simpleaudio as sa
import time
import pandas as pd
from scipy.signal import find_peaks


class TonePlayer:
    """
    This class manages sound playback based on time intervals and frequencies.
    """

    def __init__(self, sample_rate=44100):
        """
        Initializes the TonePlayer with the desired sample rate.
        """
        self.sample_rate = sample_rate
        self.current_step = 0
        self.last_play_time = time.time()

    def set_intervals(self, step_interval):
        """
        Sets the list of time intervals for sound playback.
        """
        self.step_interval = step_interval

    def set_base_pitch(self, base_pitch):
        """
        Sets the base pitch (frequency) for sound generation.
        """
        self.base_pitch = base_pitch

    def play_tone(self):
        """
        Generates and plays a sound based on the current step and base pitch.
        """
        try:
            frequency = self.base_pitch
            t = np.linspace(0, 0.2, int(self.sample_rate * 0.2), endpoint=False)
            wave = 0.5 * np.sin(2 * np.pi * frequency * t)
            audio = (wave * 32767).astype(np.int16)
            print("Playing tone:", audio[:10])  # Debug print to show a slice of the audio buffer
            sa.play_buffer(audio, 1, 2, self.sample_rate)

            self.last_play_time = time.time()
            self.current_step += 1

            if self.current_step >= len(self.step_interval):
                self.current_step = 0  # Reset step counter
        except Exception as e:
            print(f"Error in play_tone: {e}")

    def play_loop(self, step_interval, base_pitch):
        """
        Plays tones in a loop based on the provided intervals and base pitch.
        """
        self.set_intervals(step_interval)
        self.set_base_pitch(base_pitch)

        while True:
            try:
                elapsed_time = time.time() - self.last_play_time
                if elapsed_time >= self.step_interval[self.current_step]:
                    self.play_tone()
            except Exception as e:
                print(f"Error in play_loop: {e}")


def detect_peaks(accel, time, threshold, min_distance):
    stride_peaks, _ = find_peaks(accel, height=threshold, distance=min_distance)
    peak_times = time.loc[stride_peaks]
    distances = np.diff(peak_times)
    return distances


# Example usage
run = pd.read_csv("./data/pelvis_slow_labels.csv")
print(run.head())  # Debug print to check the data

player = TonePlayer()
step_interval = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
# step_interval = detect_peaks(run['acc_y'],run['timestamp'],4,10)
print("Step intervals:", step_interval)  # Debug print to show the detected intervals
base_pitch = 440

player.play_loop(step_interval, base_pitch)
