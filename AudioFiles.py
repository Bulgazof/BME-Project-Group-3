import numpy as np
import sounddevice as sd
import time
import pandas as pd
from scipy.signal import find_peaks
import threading
import os
from datetime import datetime

class TonePlayer:
    """
    This class is used for the generation of tones when the user is running, giving audio feedback.
    """

    def __init__(self, base_pitch=440, sample_rate=44100):
        """
        Initializes the TonePlayer with the desired sample rate.
        """
        self.sample_rate = sample_rate
        self.current_step = 0
        self.last_play_time = time.time()
        self.step_interval = []
        self.base_pitch = base_pitch
        self.beep_triple = 0
        self.thread = None
        self.path = "data/pelvis"
        self.data = self.fileFinder()
        self.detect_peaks()

    def play_tone(self):
        """
        Generates and plays a sound based on the current step and base pitch.
        """
        try:
            frequency = self.base_pitch
            t = np.linspace(0, 0.2, int(self.sample_rate * 0.2), endpoint=False)
            wave = 0.5 * np.sin(2 * np.pi * frequency * t)
            audio = (wave * 32767).astype(np.int16)
            sd.play(audio, self.sample_rate)
            sd.wait()  # Wait until the sound has finished playing

            self.last_play_time = time.time()
            self.current_step += 1

        except Exception as e:
            print(f"Error in play_tone: {e}")

    def play_loop(self):
        """
        Plays the tones at the interval requested
        :return:
        """
        while True:
            # The normal beeps for while leaning over
            if self.current_step < len(self.step_interval):
                elapsed_time = time.time() - self.last_play_time
                if elapsed_time >= self.step_interval[self.current_step]:  # Index over one
                    self.play_tone()
                else:
                    time.sleep(0.01)
            elif self.beep_triple < 3:
                # Beeps thrice once out of the initial stage
                self.base_pitch = 880
                self.play_tone()
                self.beep_triple += 1
            else:
                time.sleep(0.01)

    def start(self):
        """
        Threading version of the sound player
        """
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.play_loop)
            self.thread.daemon = True  # Ensure the thread exits when the main program does
            self.thread.start()

    def detect_peaks(self, threshold=4, min_distance=10):
        """
        Detect peaks and set step intervals.
        :param threshold: threshold to activate peak
        :param min_distance: distance between peaks for denoising
        """
        if isinstance(self.data, str) and self.data.endswith(".csv"):
            print("CSV FOUND")
            run = pd.read_csv(self.data)
            accel = run['acc_y']
            time = pd.to_datetime(run['timestamp'])
            stride_peaks, _ = find_peaks(accel, height=threshold, distance=min_distance)
            peak_times = time.iloc[stride_peaks].astype(int) / 10**9  # Convert to seconds
            distances = np.diff(peak_times)
            print(distances)
        else:
            distances = self.data
        self.step_interval = distances

    def fileFinder(self):
        try:
            files = os.listdir(self.path)
            # Filter out non-file entries (like directories)
            files = [f for f in files if os.path.isfile(os.path.join(self.path, f))]
            print(files)
            if not files:
                return [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

            def extract_datetime(filename):
                timestamp_str = filename[:19]  # Extract yyyy-mm-dd-hh-mm-ss
                return datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")

            sorted_files = sorted(files, key=extract_datetime, reverse=True)

            # Return the full path of the most recent file
            return os.path.join(self.path, sorted_files[0])

        except Exception as e:
            print(f"An error occurred: {e}")
            return [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
