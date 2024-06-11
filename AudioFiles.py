import numpy as np
import sounddevice as sd
import time
import pandas as pd
from scipy.signal import find_peaks
import threading
import os
import queue
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
        self.path = "data/"
        self.data = self.fileFinder()
        self.detect_peaks()
        self.readying = True
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
            if self.readying is not True:
                self.current_step += 1

        except Exception as e:
            print(f"Error in play_tone: {e}")

    def play_loop(self, scale):
        """
        Plays the tones at the interval requested.
        """
        while True:
            current_time = time.time()
            if self.readying:
                # Normal beeps while leaning over
                if current_time - self.last_play_time > 3:
                    self.base_pitch = 440
                    self.play_tone()
                    time.sleep(0.25)
                    self.play_tone()
                    time.sleep(0.25)
                    self.base_pitch = 880
                    self.play_tone()
                    self.base_pitch = 440
                    self.current_step = 0
                    self.last_play_time = current_time  # Update last play time
                    self.readying = False
            else:
                if self.current_step < len(self.step_interval):
                    elapsed_time = current_time - self.last_play_time
                    if elapsed_time >= self.step_interval[self.current_step]*scale:
                        self.play_tone()
                        self.last_play_time = current_time  # Update last play time
                        print(self.current_step)
                    else:
                        time.sleep(0.001)
                elif self.beep_triple < 3:
                    # Beeps thrice once out of the initial stage
                    self.base_pitch = 880
                    self.play_tone()
                    self.beep_triple += 1
                    self.last_play_time = current_time  # Update last play time
                else:
                    time.sleep(0.001)

    def start(self,scale):
        """
        Threading version of the sound player
        """
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.play_loop, args=(scale,))
            self.thread.daemon = True  # Ensure the thread exits when the main program does
            self.thread.start()

    def detect_peaks(self, threshold=3, min_distance=5):
        """
        Detect peaks and set step intervals.
        :param threshold: threshold to activate peak
        :param min_distance: distance between peaks for denoising
        """
        if isinstance(self.data, str) and self.data.endswith(".csv"):
            print("CSV FOUND")
            run = pd.read_csv(self.data)
            accel = run.iloc[:,1]
            time = run.iloc[:,6]
            stride_peaks, _ = find_peaks(accel, height=threshold, distance=min_distance)
            peak_times = time.iloc[stride_peaks]   # Convert to seconds
            print(peak_times)
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

            def extract_datetime(filename):
                timestamp_str = filename[:19]  # Extract yyyy-mm-dd-hh-mm-ss
                return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

            sorted_files = sorted(files, key=extract_datetime, reverse=True)
            print(sorted_files[0])
            # Return the full path of the most recent file
            return os.path.join(self.path, sorted_files[0])

        except Exception as e:
            print(f"An error occurred: {e}")
