import matplotlib.pyplot as plt
import pandas as pd
from angleCalculator import angleCalculator
def plot_angles(*angles, labels=None, title="Angles Over Time", xlabel="Time", ylabel="Angle (radians)"):
    """
    Plot given angles in the same diagram.

    :param angles: Variable number of lists containing angles to be plotted.
    :param labels: List of labels for each set of angles. Default is None.
    :param title: Title of the plot. Default is "Angles Over Time".
    :param xlabel: Label for the x-axis. Default is "Time".
    :param ylabel: Label for the y-axis. Default is "Angle (radians)".
    """
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


# Example usage
if __name__ == "__main__":
    # Load CSV file
    csv_file = "data/landmarks_2024-05-29_15-15-26.csv"

    # Initialize angle calculator
    angle_calculator = angleCalculator()

    # Calculate angles from CSV
    # print("test1")
    # lm_row = []
    # angles = angle_calculator.get_angle(csv_file, "all", True)
    # print("test2")
    # chest_angles = angles[0]
    # shin_left_angles = angles[1]
    # shin_right_angles = angles[2]
    # hip_left_angles = angles[3]
    # hip_right_angles = angles[4]
    #
    # # Plot angles
    # plot_angles(chest_angles, shin_left_angles, shin_right_angles, hip_left_angles, hip_right_angles,
    #             labels=["Chest", "Shin Left", "Shin Right", "Hip Left", "Hip Right"])
    print("This is a test")
    angles = angle_calculator.get_angle(csv_file, "chest", True)
    plot_angles(angles, labels="chest")