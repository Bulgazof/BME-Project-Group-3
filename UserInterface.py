import wx
import wx.lib.plot as wxplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks

sampling_frequency = 60  # Hz

runner_weight = 68  # Kg
var_names = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'timestamp']  # Initiate variable names
plot_var = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
acc_var_names = ['acc_x', 'acc_y', 'acc_z']
gyr_var_names = ['gyr_x', 'gyr_y', 'gyr_z']
speed_var_names = ['vel_x', 'vel_y', 'vel_z']


def load_data(filepath):
    df = pd.read_csv(filepath, names=var_names)  # Load the head data
    return df


def calc_norm(df, var_list, name):
    df[name] = np.sqrt(df[var_list[0]] ** 2 + df[var_list[1]] ** 2 + df[var_list[2]] ** 2)
    return df


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


def process_data_for_plotting(dataframe_list, speed_variables, acc_variables):
    for dataframe in dataframe_list:
        dataframe = calc_speed(dataframe, acc_variables, speed_variables)
        dataframe = calc_force(dataframe, runner_weight)
        dataframe = calc_norm(dataframe, speed_variables, 'vel_norm')
        dataframe = calc_power(dataframe)
    return dataframe_list


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


class MainFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MainFrame, self).__init__(*args, **kw, size=(1200, 800))  # Set the window size here
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)

        # Adjust BoxSizer for the top row and GridSizer for the rest
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.original_sizer = wx.GridSizer(rows=3, cols=2, hgap=20, vgap=20)

        # Example values
        reaction_time = 0.8
        finish = 18.5

        # Values to display
        self.values = {
            'Start Run': 'Start Run',
            'Stride Frequency': 'Stride Frequency',
            'Power': 'Power',
            'Acceleration': 'Acceleration',
            'Reaction Time': reaction_time,
        }

        self.create_widgets(panel, top_sizer)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(top_sizer, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(self.original_sizer, 1, wx.EXPAND | wx.ALL, 10)
        panel.SetSizer(main_sizer)
        self.Centre()

    def create_widgets(self, panel, top_sizer):
        first_row_added = False
        for label, value in self.values.items():
            if label == 'Start Run' and not first_row_added:
                display_label = f'{label}'
                btn = wx.Button(panel, label=display_label, size=(150, 100))
                btn.SetFont(wx.Font(25, wx.DEFAULT, wx.NORMAL, wx.BOLD))
                btn.SetBackgroundColour(wx.Colour(152, 251, 152))
                btn.Bind(wx.EVT_BUTTON, self.on_start)
                top_sizer.Add(btn, 1, wx.EXPAND | wx.ALL, 10)
                first_row_added = True
            elif label in ['Stride Frequency', 'Power', 'Acceleration']:
                display_label = f'{label}'
                btn = wx.Button(panel, label=display_label, size=(150, 50))
                btn.SetFont(wx.Font(20, wx.DEFAULT, wx.NORMAL, wx.BOLD))
                btn.SetBackgroundColour(wx.Colour(230, 230, 250))
                if label == 'Stride Frequency':
                    btn.Bind(wx.EVT_BUTTON, self.on_stride_frequency)

                elif label == 'Power':
                    btn.Bind(wx.EVT_BUTTON, self.on_power)
                elif label == 'Acceleration':
                    btn.Bind(wx.EVT_BUTTON, self.on_acceleration)
                self.original_sizer.Add(btn, 0, wx.EXPAND | wx.ALL, 10)
            else:
                display_label = f'{label}: {value}'
                lbl = wx.StaticText(panel, label=display_label)
                lbl.SetFont(wx.Font(20, wx.DEFAULT, wx.NORMAL, wx.BOLD))
                self.original_sizer.Add(lbl, 0, wx.ALIGN_CENTER | wx.ALL, 10)

    def on_stride_frequency(self, event):
        df_pelvis = load_data(r'../BME-Project-Group-3/data/pelvis_test.csv')
        df_pelvis_slow = load_data(r'../BME-Project-Group-3/data/pelvis_slow.csv')

        df_pelvis = calc_norm(df_pelvis, acc_var_names, 'norm')
        df_pelvis_slow = calc_norm(df_pelvis_slow, acc_var_names, 'norm')

        plot_multiple_stack([df_pelvis, df_pelvis_slow], acc_var_names, 'Stride Frequency')

    def on_power(self, event):
        df_pelvis = load_data(r'../BME-Project-Group-3/data/pelvis_test.csv')
        df_pelvis_slow = load_data(r'../BME-Project-Group-3/data/pelvis_slow.csv')

        df_pelvis = calc_norm(df_pelvis, acc_var_names, 'norm')
        df_pelvis_slow = calc_norm(df_pelvis_slow, acc_var_names, 'norm')

        dataframes = process_data_for_plotting([df_pelvis, df_pelvis_slow], speed_var_names, acc_var_names)


        power_frame = PowerFrame(None, title="Power Data", dataframes=dataframes)
        power_frame.Show()

    def on_acceleration(self, event):
        df_pelvis = load_data(r'../BME-Project-Group-3/data/pelvis_test.csv')
        df_pelvis_slow = load_data(r'../BME-Project-Group-3/data/pelvis_slow.csv')

        df_pelvis = calc_norm(df_pelvis, acc_var_names, 'norm')
        df_pelvis_slow = calc_norm(df_pelvis_slow, acc_var_names, 'norm')

        self.Hide()
        acceleration_frame = AccelerationFrame(None, title="Acceleration Data", dataframes=[df_pelvis, df_pelvis_slow])
        acceleration_frame.Show()

    def on_start(self, event):
        self.Hide()
        start_frame = StartFrame(None, title="Start Run")
        start_frame.Show()


class StartFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(StartFrame, self).__init__(*args, **kw, size=(1200, 800))

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        end_run_btn = wx.Button(panel, label="End Run", size=(500, 500))
        end_run_btn.SetFont(wx.Font(20, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        end_run_btn.SetBackgroundColour(wx.Colour(250, 128, 114))
        end_run_btn.Bind(wx.EVT_BUTTON, self.on_end_run)

        vbox.Add(end_run_btn, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        panel.SetSizer(vbox)

    def on_end_run(self, event):
        self.Hide()
        main_frame = MainFrame(None, title="Sensor Data Analysis")
        main_frame.Show()


class PowerFrame(wx.Frame):
    def __init__(self, *args, dataframes=None, **kw):
        super(PowerFrame, self).__init__(*args, **kw, size=(1200, 800))  # Set the window size here
        self.dataframes = dataframes
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)

        vbox = wx.BoxSizer(wx.VERTICAL)

        self.figure = plt.figure()
        self.canvas = wxplot.PlotCanvas(panel)

        self.plot_graph()

        vbox.Add(self.canvas, 1, wx.EXPAND)

        back_btn = wx.Button(panel, label='Back', size=(150, 50))
        back_btn.SetFont(wx.Font(20, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        back_btn.SetBackgroundColour(wx.Colour(230, 230, 250))
        back_btn.Bind(wx.EVT_BUTTON, self.on_back)

        vbox.Add(back_btn, 0, wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, 20)

        panel.SetSizer(vbox)
        self.Centre()

    def plot_graph(self):
        # Plot power graph
        data1 = [(row['timestamp'], row['power']) for _, row in self.dataframes[0].iterrows()]
        data2 = [(row['timestamp'], row['power']) for _, row in self.dataframes[1].iterrows()]

        power_data1 = wxplot.PolyLine(data1, colour='blue', legend='Run 1')
        power_data2 = wxplot.PolyLine(data2, colour='red', legend='Run 2')

        graphics = wxplot.PlotGraphics([power_data1, power_data2], "Power Data", "Time", "Power")
        self.canvas.SetEnableLegend(True)
        self.canvas.Draw(graphics)

    def on_back(self, event):
        self.Hide()
        main_frame = MainFrame(None, title="Sensor Data Analysis")
        main_frame.Show()


class AccelerationFrame(wx.Frame):
    def __init__(self, *args, dataframes=None, **kw):
        super(AccelerationFrame, self).__init__(*args, **kw, size=(1200, 800))  # Set the window size here
        self.dataframes = dataframes
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)

        vbox = wx.BoxSizer(wx.VERTICAL)

        self.figure = plt.figure()
        self.canvas = wxplot.PlotCanvas(panel)

        self.plot_graph()

        vbox.Add(self.canvas, 1, wx.EXPAND)

        back_btn = wx.Button(panel, label='Back', size=(150, 50))
        back_btn.SetFont(wx.Font(20, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        back_btn.SetBackgroundColour(wx.Colour(230, 230, 250))
        back_btn.Bind(wx.EVT_BUTTON, self.on_back)

        vbox.Add(back_btn, 0, wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, 20)

        panel.SetSizer(vbox)
        self.Centre()

    def plot_graph(self):
        # Plot acceleration graph
        data1 = [(row['timestamp'], row['acc_y']) for _, row in self.dataframes[0].iterrows()]
        data2 = [(row['timestamp'], row['acc_y']) for _, row in self.dataframes[1].iterrows()]

        acc_data1 = wxplot.PolyLine(data1, colour='blue', legend='Run 1')
        acc_data2 = wxplot.PolyLine(data2, colour='red', legend='Run 2')

        graphics = wxplot.PlotGraphics([acc_data1, acc_data2], "Acceleration Data", "Time", "Acceleration")

        self.canvas.SetEnableLegend(True)
        self.canvas.Draw(graphics)

    def on_back(self, event):
        self.Hide()
        main_frame = MainFrame(None, title="Sensor Data Analysis")
        main_frame.Show()


if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame(None, title="Sensor Data Analysis")
    frame.Show()
    app.MainLoop()
