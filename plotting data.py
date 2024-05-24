import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks

sampling_frequency = 60  # Hz

runner_weight = 68  # Kg
# Load data
var_names = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'timestamp']  # Initiate variable names
plot_var = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
acc_var_names = ['acc_x', 'acc_y', 'acc_z']
gyr_var_names = ['gyr_x', 'gyr_y', 'gyr_z']
speed_var_names = ['vel_x', 'vel_y', 'vel_z']


def load_data(filepath):
    df = pd.read_csv(filepath, names=var_names)  # Load the head data
    # df = add_time(df)
    return df


def add_time(df):
    df['timetsamp'] = np.linspace(0, 1 / sampling_frequency * len(df), num=len(df))
    return df


def plot_multiple_comb(datasets, variables, title):
    count = 1
    for df in datasets:
        for var in variables:
            plt.plot(df['timestamp'], df[var], label=f'{var}')
            count += 1
    plt.legend()
    plt.title(title)
    plt.show()


def plot_multiple_stack(datasets, variables, title):
    num_datasets = len(datasets)
    fig, axes = plt.subplots(num_datasets, 1, figsize=(10, 5 * num_datasets))  # Create subplots

    for i, (df, ax) in enumerate(zip(datasets, axes)):
        for var in variables:
            column_name = str(var)
            ax.plot(df['timestamp'], df[var], label=f'{column_name}')
        ax.set_ylabel('Value')  # Set y-label for each subplot
        ax.legend()  # Add legend for each subplot
        ax.grid(True)  # Add grid for each subplot
        ax.set_title(f'{title} - Dataset: {i + 1}')  # Set title for each subplot

    plt.xlabel('Time')  # Common x-label for all subplots
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def calc_stride_freq(df, variable, threshold_height, invert):
    # Distance=150 because 240 spm is quick sprint --> 4 Hz per leg with sampling rate = 512 Hz giving us per step
    # about 128 measuring points per step
    if invert:
        peaks = find_peaks(-df[variable], height=threshold_height, distance=120)
    else:
        peaks = find_peaks(df[variable], height=threshold_height, distance=120)
    stride_freq = len(peaks[0])
    print(f'Single leg stride frequency: {stride_freq} [steps/min]')
    return peaks


def calc_norm(df, var_list, name):
    df[name] = np.sqrt(df[var_list[0]] ** 2 + df[var_list[1]] ** 2 + df[var_list[2]] ** 2)
    return df


def calc_attenuation(tibia, head):
    closest_values = []
    tibia_peak_loc = tibia[0]
    tibia_peak_height = tibia[1]
    head_peak_loc = head[0]
    head_peak_height = head[1]
    # Match the correct head peaks to the leg peaks
    for location in tibia_peak_loc:
        closest_value_index = np.argmin(np.abs(head_peak_loc - location))
        closest_value = head_peak_loc[closest_value_index]
        closest_height = head_peak_height['peak_heights'][closest_value_index]
        closest_values.append((closest_value, closest_height))
    head_peaks_single_leg = np.array(closest_values)
    shock_att_list = []  # Initialize as an empty list
    for index in range(len(tibia_peak_loc)):
        shock_att = (1 - (head_peaks_single_leg[index, 1] / tibia_peak_height['peak_heights'][index])) * 100
        shock_att_list.append(shock_att)
    shock_att_array = np.array(shock_att_list)
    average_shock_att = np.average(shock_att_array)
    return average_shock_att


def calc_angle(acc_around, acc_support1, acc_support2):
    angle = np.arctan(acc_support2 / (np.sqrt(acc_support1 ** 2 + acc_around ** 2))) * 180 / np.pi
    return angle


def integrate_gyroscope(gyro_data, time_interval):
    orientation = np.zeros_like(gyro_data)
    angle = 0.0
    for i in range(1, len(gyro_data)):
        angle += gyro_data[i] * 180 / np.pi * time_interval  # Integration step
        orientation[i] = angle
    return orientation


def add_angles(df):
    df['acc_y_angle'] = calc_angle(df['acc_y'], df['acc_z'], df['acc_x'])
    df['gyr_y_angle'] = integrate_gyroscope(df['gyr_y'], 1 / sampling_frequency)

    frequency = 0.05
    b, a = signal.butter(2, 2 * frequency / sampling_frequency, 'highpass', output='ba')
    df['hp_gyr_y_angle'] = signal.lfilter(b, a, df['gyr_y_angle'])
    return df


def complementary_angle(df, alpha):
    df['compl_y_angle'] = alpha * (df['hp_gyr_y_angle'] + 1 / sampling_frequency) + ((1 - alpha) * df['acc_y_angle'])
    return df


#
# def run():
#     df_pelvis = load_data(r'data\test1.csv')
#     df_tibia = load_data(r'data\test2.csv')
#     # plot_multiple_stack([df_head, df_tibia], acc_var_names, 'head and tibia')
#     calc_stride_freq(df_tibia, 'acc_x', 36, invert=True)
#     df_pelvis = calc_acc_norm(df_pelvis)
#     df_tibia = calc_acc_norm(df_tibia)
#     print(f'Tibia mean norm: {df_tibia['norm'].mean()}')
#     # plot_multiple_comb([df_head, df_tibia], ['norm'], 'norms of head and tibia')
#     tibia_peaks = calc_stride_freq(df_tibia, 'norm', 42, invert=False)
#     pelvis_peaks = calc_stride_freq(df_pelvis, 'norm', 22, invert=False)
#     average_attenuation = calc_attenuation(tibia_peaks, pelvis_peaks)
#     print(f'Average shock attenuation right leg to head: {average_attenuation}%')
#     df_tibia = add_angles(df_tibia)
#     # plot_multiple_comb([df_tibia], ['acc_y_angle', 'gyr_y_angle', 'hp_gyr_y_angle'], 'angles from accel and gyro')
#     df_tibia = complementary_angle(df_tibia, 0.9)
#     plot_multiple_comb([df_tibia], ['acc_y_angle', 'hp_gyr_y_angle'], 'angles from accel and gyro')
#     plot_multiple_comb([df_tibia], ['acc_y_angle', 'compl_y_angle', 'hp_gyr_y_angle'], 'angles from accel and gyro')


def calc_force(df, mass):
    df['force'] = df['norm'] * mass
    return df


def calc_speed(df, var_names, speed_names):
    for i, (var_name, speed_name) in enumerate(zip(var_names, speed_names)):
        df[speed_name] = integrate(df[var_name], 1 / sampling_frequency)
    return df


def integrate(data, time_interval):
    integrated_data = 0
    integrated_data += data * time_interval
    return integrated_data


def calc_power(df):
    df['power'] = df['vel_norm'] * df['force']
    return df


def plot_power(dataframe_list, speed_variables, acc_variables, plot_variables, title):
    for dataframe in dataframe_list:
        dataframe = calc_speed(dataframe, acc_variables, speed_variables)
        dataframe = calc_force(dataframe, runner_weight)
        dataframe = calc_norm(dataframe, speed_variables, 'vel_norm')
        dataframe = calc_power(dataframe)
    plot_multiple_comb(dataframe_list, plot_variables, title)


def run():
    df_pelvis = load_data(r'../BME-Project-Group-3/data/pelvis.csv')
    df_tibia = load_data(r'../BME-Project-Group-3/data/tibia.csv')
    df_pelvis_slow = load_data(r'../BME-Project-Group-3/data/pelvis_slow.csv')
    df_tibia_slow = load_data(r'../BME-Project-Group-3/data/tibia_slow.csv')
    df_pelvis = calc_norm(df_pelvis, acc_var_names, 'norm')
    df_pelvis_slow = calc_norm(df_pelvis_slow, acc_var_names, 'norm')

    df_tibia = calc_norm(df_tibia, acc_var_names, 'norm')
    df_tibia_slow = calc_norm(df_tibia_slow, acc_var_names, 'norm')
    # plot_multiple_stack([df_pelvis, df_pelvis_slow], acc_var_names, 'all values')
    # plot_multiple_stack([df_tibia, df_tibia_slow], acc_var_names, 'all values')

    plot_power([df_pelvis], speed_var_names, acc_var_names, ['power', 'acc_y'],
               'Power of two runs (pelvis) + accZ')
    # plot_power([df_tibia, df_tibia_slow], speed_var_names, acc_var_names, ['power'], 'power of two runs (tibia)')


if __name__ == "__main__":
    run()
