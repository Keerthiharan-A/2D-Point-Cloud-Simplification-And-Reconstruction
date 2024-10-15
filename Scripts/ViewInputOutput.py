from PointCloudVisualizer import MainPointVisualizerApp,SecondaryPointVisualizerApp
class DualPointVisualizerApp:
    def __init__(self, file_path1, file_path2):
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.app1 = None
        self.app2 = None

    def open_windows(self):
        # Initialize the main window (tk.Tk)
        self.app1 = MainPointVisualizerApp(self.file_path1)
        self.app1.geometry("600x750+100+100")
        self.app1.load_data()

        # Initialize the secondary window (tk.Toplevel)
        self.app2 = SecondaryPointVisualizerApp(self.file_path2, master=self.app1)
        self.app2.geometry("600x750+750+100")
        self.app2.load_data()

        # Start the mainloop for the main window
        self.app1.mainloop()


# Example usage
if __name__ == "__main__":
    file_path_input = "/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset/chopper/BandNoise/chopper-01-12.5-5.xy"
    file_path_output = "/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset/dude/DistortedNoise/dude0-0.015.xy"

    # Create an instance of DualPointVisualizerApp and open both windows
    app = DualPointVisualizerApp(file_path_input,file_path_output)
    app.open_windows()