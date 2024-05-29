import pandas as pd
import numpy as np
import time
from RingBuffer import RingBuffer

class angleCalculator:
    def __init__(self):
        self.start_time = time.time()
        self.current_time = 0.0
        self.bufferSize = 5
        self.angleBuffers = {angle: RingBuffer(self.bufferSize) for angle in ['chest', 'shin_left', 'shin_right', 'hip_left', 'hip_right']}

    def get_angle(self, lm_data, which_angle: str, buffer: bool):
        """
        Calculate the specified angle from the given landmark data.

        :param lm_data: Can be a DataFrame (loaded from a CSV file) or a list/array of landmarks.
                        If it's a string, it's considered as the CSV file location.
        :param which_angle: The angle to calculate. Possible values are:
                            'chest', 'shin_left', 'shin_right', 'hip_left', 'hip_right', 'all'.
        :param buffer: Whether to use a buffer for smoothing the angle calculation.
        :return: The calculated angle or a list of angles if lm_data is a DataFrame.
        """
        if isinstance(lm_data, str):
            # Load CSV file if lm_data is a file location
            lm_data = pd.read_csv(lm_data)
            return self.process_csv(lm_data, which_angle, buffer)
        if isinstance(lm_data, pd.DataFrame):
            # Process DataFrame
            return self.process_csv(lm_data, which_angle, buffer)
        else:
            # Process list/array of landmarks
            return self.calculate_angle(lm_data, which_angle, buffer)

    def calculate_angle(self, lm_array, which_angle: str, buffer: bool):
        """
        Helper function to calculate the angle from a list/array of landmarks.
        """
        result = []
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
            case "all":
                left_chest = self.two_point_angle(lm_array, 11, 23)
                right_chest = self.two_point_angle(lm_array, 12, 24)
                chest = (left_chest + right_chest) / 2
                shin_left = self.two_point_angle(lm_array, 25, 27)
                shin_right = self.two_point_angle(lm_array, 26, 28)
                hip_left = self.three_point_angle(lm_array, 25, 23, 11)
                hip_right = self.three_point_angle(lm_array, 26, 24, 12)
                result = [chest, shin_left, shin_right, hip_left, hip_right]

        if buffer:
            if which_angle == "all":
                for i, angle_name in enumerate(['chest', 'shin_left', 'shin_right', 'hip_left', 'hip_right']):
                    self.angleBuffers[angle_name].add(result[i])
                    result[i] = self.angleBuffers[angle_name].median()
            else:
                self.angleBuffers[which_angle].add(result)
                result = self.angleBuffers[which_angle].median()

        return self.normalize_angle(result)

    def two_point_angle(self, lm_arr, pointA, pointB):
        """
        Helper function to calculate the angle between two points.
        """
        if isinstance(lm_arr, pd.Series):  # When processing a DataFrame row
            A = np.array([lm_arr[f'lm{pointA}_x'], lm_arr[f'lm{pointA}_y']])
            B = np.array([lm_arr[f'lm{pointB}_x'], lm_arr[f'lm{pointB}_y']])
        else:  # When processing a list/array of landmarks
            A = np.array([lm_arr[pointA].x, lm_arr[pointA].y])
            B = np.array([lm_arr[pointB].x, lm_arr[pointB].y])
        AB = B - A
        angle = np.arctan2(AB[1], AB[0])
        return angle

    def three_point_angle(self, lm_arr, pointA, pointB, pointC):
        """
        Helper function to calculate the angle formed by three points.
        """
        if isinstance(lm_arr, pd.Series):  # When processing a DataFrame row
            A = np.array([lm_arr[f'lm{pointA}_x'], lm_arr[f'lm{pointA}_y']])
            B = np.array([lm_arr[f'lm{pointB}_x'], lm_arr[f'lm{pointB}_y']])
            C = np.array([lm_arr[f'lm{pointC}_x'], lm_arr[f'lm{pointC}_y']])
        else:  # When processing a list/array of landmarks
            A = np.array([lm_arr[pointA].x, lm_arr[pointA].y])
            B = np.array([lm_arr[pointB].x, lm_arr[pointB].y])
            C = np.array([lm_arr[pointC].x, lm_arr[pointC].y])
        BA = A - B
        BC = C - B
        dot_product = np.dot(BA, BC)
        norm_vector1 = np.linalg.norm(BA)
        norm_vector2 = np.linalg.norm(BC)
        cos_theta = dot_product / (norm_vector1 * norm_vector2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        return angle

    def process_csv(self, df, which_angle, buffer):
        """
        Process each row of the DataFrame and calculate the specified angle.
        """
        angles = []
        for _, row in df.iterrows():
            angle = self.calculate_angle(row, which_angle, buffer)
            angles.append(angle)
        return angles

    def normalize_angle(self, angle):
        """
        Normalize an angle or a list of angles to be in the range [0, 2*pi].

        :param angle: A single angle or a list of angles.
        :return: Normalized angle(s).
        """
        if isinstance(angle, list):
            return [a % (2 * np.pi) for a in angle]
        else:
            return angle % (2 * np.pi)

def plot_angles(*angles, labels=None, title="Angles Over Time", xlabel="Time", ylabel="Angle (radians)"):
    """
    Plot given angles in the same diagram.

    :param angles: Variable number of lists containing angles to be plotted.
    :param labels: List of labels for each set of angles. Default is None.
    :param title: Title of the plot. Default is "Angles Over Time".
    :param xlabel: Label for the x-axis. Default is "Time".
    :param ylabel: Label for the y-axis. Default is "Angle (radians)".
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for i, angle_set in enumerate(angles):
        if labels and i < len(labels):
            plt.plot(angle_set, label=labels[i])
        else:
            plt.plot(angle_set, label=f'Angle Set {i + 1}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Usage example
# if __name__ == "__main__":
#     # Initialize the angle calculator
#     angle_calculator = angleCalculator()
#
#     # Calculate angles from a CSV file
#     csv_file = 'data/landmarks_2024-05-29_15-15-26.csv'
#     chest_angles = angle_calculator.get_angle(csv_file, which_angle='chest', buffer=True)
#     shin_left_angles = angle_calculator.get_angle(csv_file, which_angle='shin_left', buffer=True)
#     shin_right_angles = angle_calculator.get_angle(csv_file, which_angle='shin_right', buffer=True)
#     hip_left_angles = angle_calculator.get_angle(csv_file, which_angle='hip_left', buffer=True)
#     hip_right_angles = angle_calculator.get_angle(csv_file, which_angle='hip_right', buffer=True)
#
#     # Plot angles
#     plot_angles(chest_angles, shin_left_angles, shin_right_angles, hip_left_angles, hip_right_angles,
#                 labels=["Chest", "Shin Left", "Shin Right", "Hip Left", "Hip Right"])
if __name__ == "__main__":
    # Initialize the angle calculator
    angle_calculator = angleCalculator()

    # Calculate all angles from a CSV file
    csv_file = 'data/live_data/2024-05-29_17-26-53_landmarks.csv'
    all_angles = angle_calculator.get_angle(csv_file, which_angle='all', buffer=True)

    # Unpack the angles for plotting
    chest_angles, shin_left_angles, shin_right_angles, hip_left_angles, hip_right_angles = zip(*all_angles)

    # Plot angles
    plot_angles(chest_angles, shin_left_angles, shin_right_angles, hip_left_angles, hip_right_angles,
                labels=["Chest", "Shin Left", "Shin Right", "Hip Left", "Hip Right"])
