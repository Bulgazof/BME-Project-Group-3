import cv2
import mediapipe as mp
import math
import threading
from angleCalculator import angleCalculator

class PoseEstimator(threading.Thread):
    def __init__(self, data_queue):
        super().__init__()
        self.data_queue = data_queue
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.cap = cv2.VideoCapture(0)
        self.angle_calculator = angleCalculator()
        self.running = True

    def run(self):
        with self.mp_pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened() and self.running:
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

                lm = results.pose_landmarks
                if lm is not None:
                    lm_arr = lm.landmark

                    left_shin_angle = self.angle_calculator.get_angle(lm_arr, "shin_left", True)
                    right_shin_angle = self.angle_calculator.get_angle(lm_arr, "shin_right", True)
                    left_hip_angle = self.angle_calculator.get_angle(lm_arr, "hip_left", True)
                    right_hip_angle = self.angle_calculator.get_angle(lm_arr, "hip_right", True)
                    chest_angle = self.angle_calculator.get_angle(lm_arr, "chest", True)

                    height, width, _ = image.shape
                    zero_vector = (int(width/2), int(height/2))
                    draw_vector_coords = (zero_vector[0] - int(200 * math.cos(right_shin_angle)),
                                          zero_vector[1] - int(200 * math.sin(right_shin_angle)))
                    image = cv2.line(image, zero_vector, draw_vector_coords, (0, 255, 0), 5)

                    image = cv2.line(image, (zero_vector[0] + 100, zero_vector[1]),
                                     (int(width/2)-100, int(height/2)), (0, 0, 255), 5)
                    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

                    # Send angles to the main thread
                    self.data_queue.put({
                        'left_shin_angle': left_shin_angle,
                        'right_shin_angle': right_shin_angle,
                        'left_hip_angle': left_hip_angle,
                        'right_hip_angle': right_hip_angle,
                        'chest_angle': chest_angle
                    })

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False