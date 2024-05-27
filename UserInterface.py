import wx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

# Constants for reaction time
reaction_time = 0.8  # Reaction time in seconds
finish = 18.5  # Time to finish

# Main Frame for weight input
class WeightInputFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(WeightInputFrame, self).__init__(*args, **kw)

        # Create panel and sizer for layout
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Input field for weight
        self.weight_input = wx.TextCtrl(panel, value="")

        # Add label and input field to sizer
        sizer.Add(wx.StaticText(panel, label="Input Weight to Start:"), 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        sizer.Add(self.weight_input, 0, wx.ALL | wx.EXPAND, 5)

        # Proceed button
        proceed_button = wx.Button(panel, label="Proceed")
        proceed_button.Bind(wx.EVT_BUTTON, self.on_proceed)
        sizer.Add(proceed_button, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        panel.SetSizer(sizer)

        self.SetTitle('Input Weight')
        self.SetSize((300, 150))
        self.Centre()

    def on_proceed(self, event):
        weight = self.weight_input.GetValue()
        if weight:
            try:
                global power
                weight = float(weight)
                power = weight * 9.81  # Calculate power using weight and gravity
                self.Close()
                frame = MyFrame(None)
                frame.Show()
            except ValueError:
                wx.MessageBox("Please enter a valid number", "Error", wx.OK | wx.ICON_ERROR)

# Main Frame for application
class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)

        # Create panel and grid sizer for layout
        self.panel = wx.Panel(self)
        self.original_sizer = wx.GridSizer(rows=3, cols=2, hgap=20, vgap=20)

        # Values to display
        self.values = {
            'Key Frames': 'Key Frames',
            'Stride Frequency': 'Stride Frequency',
            'Power': 'Power',
            'Reaction Time': reaction_time,
            'Time to finish': finish,
        }

        self.create_widgets()

        self.panel.SetSizer(self.original_sizer)

        self.SetTitle('Custom UI')
        self.SetSize((400, 300))
        self.Centre()

    def create_widgets(self):
        # Create and add widgets to sizer
        for label, value in self.values.items():
            if label in ['Stride Frequency', 'Key Frames', 'Power']:
                display_label = f'{label}'
                btn = wx.Button(self.panel, label=display_label, size=(150, 50))
                btn.SetFont(wx.Font(20, wx.DEFAULT, wx.NORMAL, wx.BOLD))
                btn.SetBackgroundColour(wx.Colour(230, 230, 250))
                if label == 'Stride Frequency':
                    btn.Bind(wx.EVT_BUTTON, self.on_stride_frequency)
                elif label == 'Key Frames':
                    btn.Bind(wx.EVT_BUTTON, self.on_key_frames)
                self.original_sizer.Add(btn, 0, wx.EXPAND)
            else:
                display_label = f'{label}: {value}'
                lbl = wx.StaticText(self.panel, label=display_label)
                lbl.SetFont(wx.Font(20, wx.DEFAULT, wx.NORMAL, wx.BOLD))
                self.original_sizer.Add(lbl, 0, wx.ALIGN_CENTER)

    def plot_data(self, path, title):
        fig, ax = plt.subplots()
        pelvis_data = pd.read_csv(path)
        ax.plot(pelvis_data['timestamp'], pelvis_data['acc_x'], label='acc_x')
        ax.plot(pelvis_data['timestamp'], pelvis_data['acc_y'], label='acc_y')
        ax.plot(pelvis_data['timestamp'], pelvis_data['acc_z'], label='acc_z')
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Acceleration (g)')
        canvas = FigureCanvas(self.panel, -1, fig)
        return canvas

    def on_stride_frequency(self, event):
        # Handle Stride Frequency button click
        self.panel.DestroyChildren()
        new_sizer = wx.BoxSizer(wx.VERTICAL)
        pathPelvis = r'pelvis.csv'
        # Plotting pelvis accelerometer data
        canvasPelvis = self.plot_data(pathPelvis, "Pelvis Accelerometer Data")

        # Back button
        back_button = wx.Button(self.panel, label='Go Back', size=(150, 50))
        back_button.Bind(wx.EVT_BUTTON, self.on_go_back)

        # Add plot and back button to sizer
        new_sizer.Add(canvasPelvis, 1, wx.EXPAND)
        new_sizer.Add(back_button, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        self.panel.SetSizer(new_sizer)
        self.panel.Layout()

    def on_key_frames(self, event):
        # Handle Key Frames button click
        self.panel.DestroyChildren()
        new_sizer = wx.BoxSizer(wx.VERTICAL)

        key_frame_text = wx.StaticText(self.panel, label="Key Frames: Displaying Key Frame Data Here")
        key_frame_text.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD))

        # Back button
        back_button = wx.Button(self.panel, label='Go Back', size=(150, 50))
        back_button.Bind(wx.EVT_BUTTON, self.on_go_back)

        # Add key frame text and back button to sizer
        new_sizer.Add(key_frame_text, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        new_sizer.Add(back_button, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        self.panel.SetSizer(new_sizer)
        self.panel.Layout()

    def on_go_back(self, event):
        # Handle Go Back button click
        self.panel.DestroyChildren()
        self.original_sizer = wx.GridSizer(rows=3, cols=2, hgap=20, vgap=20)
        self.create_widgets()
        self.panel.SetSizer(self.original_sizer)
        self.panel.Layout()

# Main Application
class MyApp(wx.App):
    def OnInit(self):
        input_frame = WeightInputFrame(None)
        input_frame.Show(True)
        return True

def start_ui():
    app = MyApp()
    app.MainLoop()


if __name__ == '__main__':
    app = MyApp()
    app.MainLoop()

