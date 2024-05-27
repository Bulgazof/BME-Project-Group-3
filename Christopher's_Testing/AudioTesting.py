from AudioFiles import TonePlayer
import time

player = TonePlayer()
player.detect_peaks([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
# player.detect_peaks(".././data/pelvis_slow_labels.csv")
player.base_pitch = 440

# Calls threading
player.start()

while True:
    time.sleep(1)
    player.base_pitch = player.base_pitch + 40
    print("Main program running...")
