import numpy as np
import csv
from scipy.signal import find_peaks
from CreaTeBME import SensorManager
import time
from datetime import datetime
import threading
import sys

class RunnerIMU:
    FREQUENCY = 60
    BUFFER_SIZE = FREQUENCY * 5

    def __init__(self, pelvis_sensor, duration, steps):
        self.SENSORS_NAMES = [pelvis_sensor]
        self.running = False
        self.step_num = steps
        self.record_duration = duration
        self.t_stamp = {name: np.array([0.0]) for name in self.SENSORS_NAMES}
        self.acc = {name: np.empty((0, 3)) for name in self.SENSORS_NAMES}
        self.gyr = {name: np.empty((0, 3)) for name in self.SENSORS_NAMES}
        self.data = {name: np.empty((0, 7)) for name in self.SENSORS_NAMES}
        self.sensors_connected = False

    def init_sensors(self):
        '''

        :return: True when finished and connected. False when finished and not connected
        '''
        try:
            self.manager = SensorManager(self.SENSORS_NAMES)
            self.manager.set_sample_rate(self.FREQUENCY)
            self.manager.start()
            return True
        except ConnectionError:
            return False
        #     when done return True
        # if fails, return False

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

    def detect_peaks(self, accel, time, threshold, min_distance):
        stride_peaks, _ = find_peaks(accel, height=threshold, distance=min_distance)
        peak_times = time[stride_peaks]
        distances = np.diff(peak_times)
        print(peak_times)
        return peak_times

    def save_to_csv(self, sensor_name, file_path, peaks):
        if len(peaks) > 0:
            desired_peak = peaks[self.step_num]
            num_rows_to_save = int(desired_peak * self.FREQUENCY)
        else:
            num_rows_to_save = len(self.data)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in self.data[sensor_name][:num_rows_to_save]:
                writer.writerow(row)

    def run_analysis(self):
        time.sleep(self.record_duration)
        if not self.running:
            return
        self.update_measurements()
        pelvis_y_accel = self.acc[self.SENSORS_NAMES[0]][:, 1]
        peaks = self.detect_peaks(pelvis_y_accel, self.t_stamp[self.SENSORS_NAMES[0]], 4, 10)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("Updating measurements and saving to CSV...")
        self.save_to_csv('FD92', f'data/live_data/{current_time}_pelvis.csv', peaks)
        self.save_to_csv('F30E', f'data/live_data/{current_time}_tibia.csv', peaks)
        self.running = False  # Stop the function after updating and saving

    def record(self):
        if not self.running:
            print("Starting the recording...")
            self.manager._clear_queue()
            self.running = True
            self.run_analysis()

    def close(self):
        self.manager.stop()
        sys.exit()

    def listen_to_queue(self):
        while True:
            command = global_queue.get()
            if command == "START_IMU_SETUP":
                print("Got Setup in IMU")
                if self.init_sensors():  # get stuck on this while initializing
                    print("sensors initialized")
                    self.sensors_connected = True
                    global_queue.put("IMU_SETUP_FINISHED")
                else:
                    print("sensors initialization timed out")
                    global_queue.put("IMU_SETUP_FAILED")
            elif command == "START_IMU_RECORD" and self.sensors_connected:
                print("Got Record in IMU")
                self.record()
            elif command == "IMU_CLOSE" and self.sensors_connected:
                print("Got Close in IMU")
                self.close()
            else:
                global_queue.put(command)
                time.sleep(0.1)  # This should be outside the else block to avoid starvation

def start_IMU(queue):
    global global_queue
    global_queue = queue
    frame = RunnerIMU('F30E', 5, 10)
    imu_thread = threading.Thread(target=frame.listen_to_queue)
    imu_thread.start()
    print("IMU thread started")