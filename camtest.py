import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math
from angleCalculator import angleCalculator
from RingBuffer import RingBuffer


start_time = time.time()  # Get the current time
condition_met = False  # Initialize the condition flag
current_time = 0.0

squatTrig = False

squatCount = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pygame.init()
cap = cv2.VideoCapture(0)
angle_calculator = angleCalculator()

with mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Flip the image horizontally for a selfie-view display.
    lm = results.pose_landmarks
    if lm is not None:
        lm_arr = lm.landmark

        left_shin_angle = angle_calculator.get_angle(lm_arr, "shin_left", True)
        right_shin_angle = angle_calculator.get_angle(lm_arr, "shin_right", True)
        left_hip_angle = angle_calculator.get_angle(lm_arr, "hip_left", True)
        right_hip_angle = angle_calculator.get_angle(lm_arr, "hip_right", True)
        chest_angle = angle_calculator.get_angle(lm_arr, "chest", True)

        # TODO Fix or just remove this
        # angleBuffer.add(temp_angle)
        # lefttHsBuffer.add(temp_left_hs)
        # rightHsBuffer.add(temp_right_hs)
        #
        # angle = np.mean(angleBuffer.get())
        # left_hs = np.mean(leftHsBuffer.get(), axis=0)
        # right_hs = np.mean(rightHsBuffer.get(), axis=0)

        # draw the vector
        height, width, _ = image.shape
        zero_vector = (int(width/2), int(height/2)) # vector that points to middle of screen to draw other vectors
        print(right_shin_angle)
        draw_vector_coords = (zero_vector[0] - int(200 * math.cos(right_shin_angle)), zero_vector[1] - int(200*math.sin(right_shin_angle)))
        image = cv2.line(image, zero_vector, draw_vector_coords, (0, 255, 0), 5)

        #straight up plotting the coordinates found from the program, no angle calulcation
        # image = cv2.line(image, zero_vector, (int(width/2) - int(temp_left_hs[0]*200), int(height/2) - int(temp_left_hs[1]*200)), (255, 0, 0), 10)

        #this line is a horizontal plane on the screen
        image = cv2.line(image, (zero_vector[0] + 100, zero_vector[1]), (int(width/2)-100, int(height/2)), (0, 0, 255), 5)
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
      break
    
pygame.quit()
cap.release