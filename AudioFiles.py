import numpy as np
import sounddevice as sd
import time
import pandas as pd
from scipy.signal import find_peaks
import threading

class TonePlayer:
    """
    This class is used for the generation of tones when the user is running, giving audio feedback.
    """

    def __init__(self,data = 'none', base_pitch = 440, sample_rate=44100):
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
        self.data = data
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
                if elapsed_time >= self.step_interval[self.current_step]: # Index over one
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

    def detect_peaks(self,threshold=4, min_distance=10):
        """
        Detect peaks and set step intervals.
        :param csv: string path to the csv file
        :param threshold: threshold to activate peak
        :param min_distance: distance between peaks for denoising
        """
        if type(self.data) == str:
            run = pd.read_csv(self.data)
            accel = run['acc_y']
            time = run['timestamp']
            stride_peaks, _ = find_peaks(accel, height=threshold, distance=min_distance)
            peak_times = time.loc[stride_peaks]
            distances = np.diff(peak_times)
        else:
            distances = self.data
        self.step_interval = distances
