import threading
import time
from UserInterface import start_ui
from camtest import CameraEstimator
from AudioFiles import TonePlayer


audio = TonePlayer([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

if __name__ == "__main__":
    # Start the UI thread
    ui_thread = threading.Thread(target=start_ui)
    ui_thread.daemon = False  # Allows the program to exit even if the thread is still running
    ui_thread.start()

    # Start the camera thread
    camera_thread = threading.Thread(target=CameraEstimator.camera_start)
    camera_thread.start()