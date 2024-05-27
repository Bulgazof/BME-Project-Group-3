import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math
import csv
import os

from angleCalculator import angleCalculator
from AudioFiles import TonePlayer
from RingBuffer import RingBuffer

# Initialize variables
start_time = time.time()  # Get the current time
condition_met = False  # Initialize the condition flag
current_time = 0.0
frame = 0
display_FPS = 0.5
prev_time = time.time()
cum_frame_time = 0
prev_frame_time = time.time()

# Put before while loop
player_1 = TonePlayer([2, 2, 2, 2, 2, 2, 2, 2, 2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],440)
player_1.start()


# Initialize MediaPipe Pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Initialize Pygame and video capture
pygame.init()
cap = cv2.VideoCapture(0)

# Initialize angle calculator
angle_calculator = angleCalculator()

# Function to save landmark array to a CSV file
def save_landmarks_to_csv(lm_arr, filename="C:/Users/othoe/PycharmProjects/SprintMeister/landmarks.csv"):
    # Check if the file exists
    file_exists = os.path.exists(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file doesn't exist
        if not file_exists:
            header = []
            for i in range(len(lm_arr)):
                header.extend(['lm{}_x'.format(i + 1), 'lm{}_y'.format(i + 1), 'lm{}_z'.format(i + 1)])
            writer.writerow(header)

        # Write landmark data
        row = []
        for lm in lm_arr:
            row.extend([lm.x, lm.y, lm.z])  # Append x, y, z coordinates of each landmark
        writer.writerow(row)


# Initialize video writer object
output_filename = 'C:/Users/othoe/PycharmProjects/SprintMeister/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))

# Start MediaPipe Pose estimation
with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
        cum_frame_time += 1/(time.time() - prev_frame_time)
        print(1/(time.time() - prev_frame_time))
        prev_frame_time = time.time()

        # Convert the image back to BGR for OpenCV
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks on the image
        if results.pose_landmarks:
            # mp_drawing.draw_landmarks(
            #     image,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            # )

            # Get the landmarks
            lm_arr = results.pose_landmarks.landmark

            # image = np.zeros_like(image)

            # Calculate angles
            chest_angle = angle_calculator.get_angle(lm_arr, "chest", True)
            print(chest_angle)
            player_1.base_pitch = 440 + (chest_angle - 1) * 80
            # Draw vector for right shin angle
            height, width, _ = image.shape
            zero_vector = (int(width / 2), int(height / 2))  # Center of the screen
            draw_vector_coords = (
                zero_vector[0] - int(200 * math.cos(chest_angle)),
                zero_vector[1] - int(200 * math.sin(chest_angle))
            )
            image = cv2.line(image, zero_vector, draw_vector_coords, (0, 255, 0), 5)

            # Draw a horizontal line on the screen
            image = cv2.line(image, (zero_vector[0] + 100, zero_vector[1]), (int(width / 2) - 100, int(height / 2)), (0, 0, 255), 5)

            # Save landmarks to CSV
            save_landmarks_to_csv(lm_arr)

        # Write frame to output video
        output_video.write(image)

        # Display the image
        if prev_time + 1/display_FPS < time.time():
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            prev_time = time.time()

        # Break the loop on 'ESC' key press
        if cv2.waitKey(5) & 0xFF == 27:
            print(f"Average Frame Rate: {cum_frame_time/frame}")
            break

# Release resources
pygame.quit()
cap.release()
output_video.release()
cv2.destroyAllWindows()
