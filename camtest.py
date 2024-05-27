import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os

from angleCalculator import angleCalculator

# Initialize variables
start_time = time.time()
condition_met = False
current_time = 0.0
frame = 0
frame_time = 0
cum_fps = 0
display_FPS = 0.5
prev_time = time.time()

# Initialize MediaPipe Pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize angle calculator
angle_calculator = angleCalculator()

# Function to save landmark array to a CSV file
def save_landmarks_to_csv(lm_arr, filename="C:/Users/othoe/PycharmProjects/SprintMeister/landmarks.csv"):
    file_exists = os.path.exists(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            header = []
            for i in range(len(lm_arr)):
                header.extend(['lm{}_x'.format(i + 1), 'lm{}_y'.format(i + 1), 'lm{}_z'.format(i + 1)])
            writer.writerow(header)
        row = []
        for lm in lm_arr:
            row.extend([lm.x, lm.y, lm.z])
        writer.writerow(row)

# Initialize video writer object
output_filename = 'C:/Users/othoe/PycharmProjects/SprintMeister/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))

# Start MediaPipe Pose estimation
with mp_pose.Pose(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image to RGB and process it with MediaPipe Pose
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        frame += 1
        x = 1 / (time.time() - frame_time)
        cum_fps += x
        frame_time = time.time()

        # If pose landmarks are detected
        if results.pose_landmarks:
            lm_arr = results.pose_landmarks.landmark

            # Save selected landmarks to CSV (example: landmarks 0, 11, 12, 23, 24)
            selected_landmarks = [lm_arr[i] for i in [11, 12, 23, 24, 26, 27, 28]]
            save_landmarks_to_csv(selected_landmarks)

        # Write frame to output video
        output_video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Display the image at a reduced frame rate
        if prev_time + 1 / display_FPS < time.time():
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            prev_time = time.time()

        # Break the loop on 'ESC' key press
        if cv2.waitKey(5) & 0xFF == 27:
            print(f"Average Frame Rate: {cum_fps / frame}")
            break

# Release resources
print(f"Average Frame Rate: {cum_fps / frame}")
cap.release()
output_video.release()
cv2.destroyAllWindows()