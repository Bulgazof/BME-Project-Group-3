import threading
import time
from UserInterface import start_ui

def camera_func():
    # Simulate camera processing


if __name__ == "__main__":
    # Start the UI thread
    ui_thread = threading.Thread(target=start_ui)
    ui_thread.daemon = False  # Allows the program to exit even if the thread is still running
    ui_thread.start()


    # Start the camera thread
    camera_thread = threading.Thread(target=camera_func)
    camera_thread.start()