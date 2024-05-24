from RingBuffer import RingBuffer
import time
import numpy as np


class angleCalculator:
    def __init__(self):

        self.start_time = time.time()  # Get the current time
        self.current_time = 0.0
        self.bufferSize = 3

        self.angleBuffer = RingBuffer(self.bufferSize)

    def get_angle(self, lm_array, which_angle: str, buffer: bool):
        '''
        :param which_angle: What angle do you want? possible values are
        chest (average angle of left and right chest to ground)
        shin_left / shin_right (left, right)
        hip_left / hip_right --> upper leg to chest(left, right)
        '''
        result = 0
        match which_angle:
            case "chest":
                left_chest = self.two_point_angle(lm_array, 11, 23)
                right_chest = self.two_point_angle(lm_array, 12, 24)
                result = (left_chest + right_chest) / 2
            case "shin_left":
                result = self.two_point_angle(lm_array, 25, 27)
            case "shin_right":
                result = self.two_point_angle(lm_array, 26, 28)
            case "hip_left":
                result = self.three_point_angle(lm_array, 25, 23, 11)
            case "hip_right":
                result = self.three_point_angle(lm_array, 26, 24, 12)
        if buffer:
            self.angleBuffer.add(result)
            result = self.angleBuffer.median()
            return result
        else:
            return result

    def two_point_angle(self, lm_arr, pointA, pointB):
        A = np.array([lm_arr[pointA].x, lm_arr[pointA].y])
        B = np.array([lm_arr[pointB].x, lm_arr[pointB].y])

        AB = (B - A)

        angle = (np.arctan2(AB[1], AB[0]))
        return angle

    def three_point_angle(self, lm_arr, pointA, pointB, pointC):
        A = np.array([lm_arr[pointA].x, lm_arr[pointA].y])
        B = np.array([lm_arr[pointB].x, lm_arr[pointB].y])
        C = np.array([lm_arr[pointC].x, lm_arr[pointC].y])

        BA = (A - B)
        BC = (C - B)
        # Calculate the dot product
        dot_product = np.dot(BA, BC)

        # Calculate the magnitudes (norms) of the vectors
        norm_vector1 = np.linalg.norm(BA)
        norm_vector2 = np.linalg.norm(BC)

        # Calculate the cosine of the angle
        cos_theta = dot_product / (norm_vector1 * norm_vector2)

        # Handle possible floating-point precision issues
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Calculate the angle in radians
        angle = np.arccos(cos_theta)

        return angle