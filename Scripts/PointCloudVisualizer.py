import tkinter as tk
from tkinter import Label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class BasePointVisualizerApp:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.file_name = file_path.split('/')[-1] if file_path else ""
        self.data = None

        load_button = tk.Button(self, text="Load Data", command=self.load_data)
        load_button.pack(pady=10)

        flip_x_button = tk.Button(self, text="Flip X Axis", command=self.flip_x_axis)
        flip_x_button.pack(pady=10)

        flip_y_button = tk.Button(self, text="Flip Y Axis", command=self.flip_y_axis)
        flip_y_button.pack(pady=10)

        self.info_label = Label(self, text="No file loaded.")
        self.info_label.pack(pady=10)

        self.setup_figure()
        
        if self.file_path:
            self.load_data()

    def setup_figure(self):
        """Setup the matplotlib figure and canvas."""
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        if self.file_path:
            try:
                self.data = np.loadtxt(self.file_path, delimiter=' ')
                self.update_info()
                self.plot_data()
            except Exception as e:
                self.info_label.config(text=f"Failed to load data: {e}")

    def plot_data(self):
        """Clear the current plot and plot new data."""
        if self.data is not None:
            self.reset_figure()
            self.ax.clear()
            self.ax.scatter(self.data[:, 0], self.data[:, 1], color='blue', s=15)
            #self.ax.set_title('2D Points')
            #self.ax.set_xlabel('X coordinate')
            #self.ax.se
            # t_ylabel('Y coordinate')
            self.ax.set_axis_off()
            self.canvas.draw_idle()

    def update_info(self):
        """Update the info label with the current file and point count."""
        if self.data is not None:
            count = len(self.data)
            self.info_label.config(text=f"File: {self.file_name} | Number of points: {count}")

    def flip_x_axis(self):
        """Flip the points along the X-axis and replot."""
        if self.data is not None:
            self.data[:, 0] = -self.data[:, 0]
            self.plot_data()

    def flip_y_axis(self):
        """Flip the points along the Y-axis and replot."""
        if self.data is not None:
            self.data[:, 1] = -self.data[:, 1]
            self.plot_data()
            
    def reset_figure(self):
        """Reset the figure and canvas."""
        self.canvas_widget.destroy()
        self.toolbar.destroy()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)


class MainPointVisualizerApp(BasePointVisualizerApp, tk.Tk):
    def __init__(self, file_path=None):
        tk.Tk.__init__(self)  # Properly initialize tk.Tk
        BasePointVisualizerApp.__init__(self, file_path)
        self.title('2D Point Visualizer - Main')
        self.geometry('600x750')


class SecondaryPointVisualizerApp(BasePointVisualizerApp, tk.Toplevel):
    def __init__(self, file_path=None, master=None):
        tk.Toplevel.__init__(self, master=master)
        BasePointVisualizerApp.__init__(self, file_path)
        self.title('2D Point Visualizer - Secondary')
        self.geometry('600x750')
        # self.transient(master)  # Comment out or remove this line


if __name__ == "__main__":
    app = MainPointVisualizerApp("/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset/apple/DistortedNoise/apple-1-0.01.xy")
    app.mainloop()