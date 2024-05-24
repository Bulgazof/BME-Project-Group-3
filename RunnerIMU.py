import numpy as np
from pynput.keyboard import Key, Listener
import threading
import csv
from CreaTeBME import SensorManager


class RunnerIMU:
    FREQUENCY = 60
    BUFFER_SIZE = FREQUENCY * 5

    def __init__(self, pelvis_sensor, tibia_sensor):
        self.SENSORS_NAMES = [pelvis_sensor, tibia_sensor]
        self.manager = SensorManager(self.SENSORS_NAMES)
        self.manager.set_sample_rate(self.FREQUENCY)
        self.manager.start()
        self.running = False
        self.thread = None
        self.t_stamp = {name: np.array([0.0]) for name in self.SENSORS_NAMES}
        self.acc = {name: np.empty((0, 3)) for name in self.SENSORS_NAMES}
        self.gyr = {name: np.empty((0, 3)) for name in self.SENSORS_NAMES}
        self.data = {name: np.empty((0, 7)) for name in self.SENSORS_NAMES}

    def update_measurements(self):
        measurements = self.manager.get_measurements()
        for name, sensor_data in measurements.items():
            if not sensor_data:
                continue
            for element in sensor_data:
                new_timestamp = self.t_stamp[name][-1] + 1.0 / self.FREQUENCY
                new_row = np.array(list(element[:6]) + [new_timestamp]).reshape(1, 7)
                self.data[name] = np.vstack((self.data[name], new_row))
                self.acc[name] = np.vstack((self.acc[name][-self.BUFFER_SIZE:], element[:3]))
                self.gyr[name] = np.vstack((self.gyr[name][-self.BUFFER_SIZE:], element[3:6]))
                self.t_stamp[name] = np.append(self.t_stamp[name][-self.BUFFER_SIZE:], new_timestamp)

    def save_to_csv(self, sensor_name, file_path):
        header = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "timestamp"]
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for row in self.data[sensor_name]:
                writer.writerow(row)

    def run_functions(self):
        while self.running:
            self.update_measurements()
            self.save_to_csv(self.SENSORS_NAMES[0], 'data/pelvis_test.csv')
            self.save_to_csv(self.SENSORS_NAMES[1], 'data/tibia_test.csv')

    def toggle_functions(self):
        if self.running:
            print("Recording stopped.")
            self.running = False
            if self.thread is not None:
                self.thread.join()
        else:
            print("Recording...")
            self.manager._clear_queue()
            self.running = True
            self.thread = threading.Thread(target=self.run_functions)
            self.thread.start()

    def on_press(self, key):
        if key == Key.space:
            self.toggle_functions()

    def toggle_record(self):
        with Listener(on_press=self.on_press) as listener:
            listener.join()

