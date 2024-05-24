import numpy as np
from pynput.keyboard import Key, Listener
import threading
import csv
from CreaTeBME import SensorManager

running = False
thread = None

FREQUENCY = 60
BUFFER_SIZE = FREQUENCY * 5
SENSORS_NAMES = ['FD92', 'F30E']

manager = SensorManager(SENSORS_NAMES)
manager.set_sample_rate(FREQUENCY)
manager.start()

t_stamp = {name: np.array([0.0]) for name in SENSORS_NAMES}
acc = {name: np.empty((0, 3)) for name in SENSORS_NAMES}
gyr = {name: np.empty((0, 3)) for name in SENSORS_NAMES}
data = {name: np.empty((0, 7)) for name in SENSORS_NAMES}

def update_measurements():
    measurements = manager.get_measurements()
    for name, sensor_data in measurements.items():
        if not sensor_data:
            continue
        for element in sensor_data:
            new_timestamp = t_stamp[name][-1] + 1.0 / FREQUENCY
            new_row = np.array(list(element[:6]) + [new_timestamp]).reshape(1, 7)
            data[name] = np.vstack((data[name], new_row))
            acc[name] = np.vstack((acc[name][-BUFFER_SIZE:], element[:3]))
            gyr[name] = np.vstack((gyr[name][-BUFFER_SIZE:], element[3:6]))
            t_stamp[name] = np.append(t_stamp[name][-BUFFER_SIZE:], new_timestamp)


def save_to_csv(sensor_name, file_path):
    header = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "timestamp"]
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in data[sensor_name]:
            writer.writerow(row)


def run_functions():
    global running
    while running:
        print("Updating measurements and saving to CSV...")
        update_measurements()
        save_to_csv('FD92', 'pelvis_slow.csv')
        save_to_csv('F30E', 'tibia_slow.csv')


def toggle_functions():
    global running, thread
    if running:
        print("Stopping the functions...")
        running = False
        if thread is not None:
            thread.join()
    else:
        print("Starting the functions...")
        manager._clear_queue()
        running = True
        thread = threading.Thread(target=run_functions)
        thread.start()


def on_press(key):
    if key == Key.space:
        toggle_functions()


def on_release(key):
    if key == Key.esc:
        return False


# Collect events until released
with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()


