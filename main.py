import threading
import time
from UserInterface import start_ui
from camtest import camera_start
from queue import Queue
from RunnerIMU import start_IMU


if __name__ == "__main__":
    big_queue = Queue(maxsize=0)

    # Start the UI thread
    ui_thread = threading.Thread(target=start_ui, args=(big_queue, ))
    ui_thread.start()

    # Start the camera thread
    camera_thread = threading.Thread(target=camera_start, args=(big_queue, ))
    camera_thread.start()

    # Start the IMU thread
    imu_thread = threading.Thread(target=start_IMU, args=(big_queue,))
    imu_thread.start()

    ui_thread.join() # only stop main thread when ui thread stops.
