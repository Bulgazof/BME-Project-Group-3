from AudioFiles import *
# Example usage
player = TonePlayer()
player.detect_peaks(".././data/pelvis_slow_labels.csv")
player.base_pitch = 440

# Start the loop
player.play_loop()