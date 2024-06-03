import cv2
import mediapipe as mp
import time
import math
import csv
import os
from datetime import datetime
from angleCalculator import angleCalculator
from AudioFiles import TonePlayer
import threading
import queue

class Camera:
    def __init__(self):
        # Initialize time and frame counters
        self.start_time = time.time()
        self.condition_met = False
        self.current_time = 0.0
        self.frame = 0
        self.display_FPS = 0.5
        self.prev_time = time.time()
        self.cum_frame_time = 0
        self.prev_frame_time = time.time()

        # Initialize tone player with frequency adjustments
        self.player_1 = TonePlayer([2, 2, 2, 2, 2, 2, 2, 2, 2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], 440)

        # Initialize MediaPipe pose components
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        self.cap = cv2.VideoCapture(0)

        # Initialize angle calculator
        self.angle_calculator = angleCalculator()

        # Setup file names for video and CSV outputs
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_filename = f'data/live_data/{current_time}_recording.mp4'
        self.csv_filename = f'data/live_data/{current_time}_landmarks.csv'

        # Initialize video writer
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(self.output_filename, self.fourcc, 20.0, (640, 480))

    def save_landmarks_to_csv(self, lm_arr):
        # Check if CSV file already exists
        file_exists = os.path.exists(self.csv_filename)
        # Open the CSV file in append mode
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write header if file does not exist
            if not file_exists:
                header = []
                for i in [11, 12, 23, 24, 25, 26, 27, 28]:
                    header.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z'])
                writer.writerow(header)
            # Write the selected landmarks to the CSV file
            row = []
            for i in [11, 12, 23, 24, 25, 26, 27, 28]:
                lm = lm_arr[i]
                row.extend([lm.x, lm.y, lm.z])
            writer.writerow(row)

    def process_frame(self, pose, image):
        # Convert image to RGB and process with MediaPipe
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convert image back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.frame += 1
        self.cum_frame_time += 1 / (time.time() - self.prev_frame_time)
        self.prev_frame_time = time.time()

        if results.pose_landmarks:
            lm_arr = results.pose_landmarks.landmark
            chest_angle = self.angle_calculator.get_angle(lm_arr, "chest", True)

            # Adjust tone based on chest angle
            self.player_1.base_pitch = 440 + (chest_angle - 1) * 80

            # Draw vector on image
            height, width, _ = image.shape
            zero_vector = (int(width / 2), int(height / 2))
            draw_vector_coords = (
                zero_vector[0] - int(200 * math.cos(chest_angle)),
                zero_vector[1] - int(200 * math.sin(chest_angle))
            )
            image = cv2.line(image, zero_vector, draw_vector_coords, (0, 255, 0), 5)
            image = cv2.line(image, (zero_vector[0] + 100, zero_vector[1]), (int(width / 2) - 100, int(height / 2)),
                             (0, 0, 255), 5)

            # Save selected landmarks to CSV
            self.save_landmarks_to_csv(lm_arr)

        # Write the frame to the video file
        self.output_video.write(image)

        # Display the frame at a set FPS
        if self.prev_time + 1 / self.display_FPS < time.time():
            cv2.imshow('Running', cv2.flip(image, 1))
            self.prev_time = time.time()

    def run(self):
        # Start the tone player
        self.player_1.start()

        # Setup MediaPipe Pose
        with self.mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                self.process_frame(pose, image)
                if cv2.waitKey(5) & 0xFF == 27:
                    print(f"Average Frame Rate: {self.cum_frame_time / self.frame}")
                    break
        self.cleanup()

    def setup(self):
        # Initial setup to display camera feed
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            else:
                cv2.imshow('Camera Setup Window', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    global_queue.put("CAMERA_SETUP_FINISHED")
                    break
            try:
                command = global_queue.get(False)
                if command == "STOP_CAMERA_SETUP":
                    print("Got Stop Setup in Camera")
                    break
                else:
                    global_queue.put(command)
            except queue.Empty:
                pass
        cv2.destroyAllWindows()

    def cleanup(self):
        # Cleanup resources
        self.cap.release()
        self.output_video.release()
        cv2.destroyAllWindows()

    def wait_for_setup(self):
        while True:
            command = global_queue.get()
            if command == "START_CAMERA_SETUP":
                print("Got Setup in Camera")
                self.setup()
            elif command == "STOP_CAMERA_SETUP":
                print("Got Stop Setup in Camera")
            else:
                global_queue.put(command)
                time.sleep(0.1)
                # global_queue.put("CAMERA_SETUP_FINISHED")
    def wait_for_record(self):
        while True:
            command = global_queue.get()
            if command == "START_RECORD":
                print("Got Record in Camera")
                self.run()
            else:
                global_queue.put(command)
                time.sleep(0.1)


if __name__ == "__main__":
    cam1 = Camera()
    cam1.setup()
    cam1.run()

def camera_start(queue):
    print("Initializing Camera...")
    global global_queue
    global_queue = queue
    camera = Camera()
    camera_setup_thread = threading.Thread(target=camera.wait_for_setup)
    camera_setup_thread.start()
    print("Camera setup thread started")
    camera_recording_thread = threading.Thread(target=camera.wait_for_record)
    camera_recording_thread.start()
    print("Camera recording thread started")
    print("Camera initialization complete")