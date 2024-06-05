from AudioFiles import TonePlayer
import time

player = TonePlayer()
# player.detect_peaks(".././data/2024-06-02-14-12-03pelvis_slow_labels.csv")


# Calls threading
player.start()
while True:
    time.sleep(0.1)

