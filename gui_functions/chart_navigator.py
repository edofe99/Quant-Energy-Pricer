import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import ImageGrab, Image

class ChartNavigator(ttk.Frame):
    
    def __init__(self, parent, figures):
        super().__init__(parent)

        self.figures = figures
        self.current_index = 0

        # Navigation buttons
        btn_frame = ttk.Frame(self)
        # btn_frame.pack(pady=5)
        btn_frame.pack(side=tk.TOP, pady=5)
        btn_frame.configure(height=50)  # Force it to take up space
        # btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Create a frame for the chart
        self.chart_frame = ttk.Frame(self)
        self.chart_frame.pack(expand=True, fill=tk.BOTH)

        # Ensure self.figures is a list, even if it's a single figure
        if not isinstance(self.figures, (list, tuple)):  
            self.figures = [self.figures]  # Convert single figure to a list
        
        if len(self.figures) >1:
            # plot back and forward buttons only if more than 1 chart
            self.btn_back = ttk.Button(btn_frame, text="←", command=self.prev_chart, state=tk.DISABLED)
            self.btn_back.pack(side=tk.LEFT, padx=5)

            self.btn_forward = ttk.Button(btn_frame, text="→", command=self.next_chart)
            self.btn_forward.pack(side=tk.RIGHT, padx=5)

        self.btn_save = ttk.Button(btn_frame, text="💾 Save", command=self.save_chart)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # Display the first chart
        self.canvas = None
        self.show_chart(self.current_index)

    def show_chart(self, index):
        """ Display the chart at the given index. """
        if self.canvas:
            self.canvas.get_tk_widget().destroy()  # Remove the previous canvas
        
        fig = self.figures[index]  # Get the current figure
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        self.canvas.draw()

        # Enable/Disable navigation buttons
        if len(self.figures) >1:
            self.btn_back.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
            self.btn_forward.config(state=tk.NORMAL if index < len(self.figures) - 1 else tk.DISABLED)

    def prev_chart(self):
        """ Navigate to the previous chart. """
        if self.current_index > 0:
            self.current_index -= 1
            self.show_chart(self.current_index)

    def next_chart(self):
        """ Navigate to the next chart. """
        if self.current_index < len(self.figures) - 1:
            self.current_index += 1
            self.show_chart(self.current_index)

    def save_chart(self):
        """ Save the current chart as a PNG file. """
        if not self.canvas:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All Files", "*.*")],
            title="Save Chart As"
        )
        if file_path:
            self.figures[self.current_index].savefig(file_path, format="png", dpi=300)
            print(f"Chart saved as {file_path}!")

    def close_navigator(self):
        """Remove all buttons, charts, and frames from ChartNavigator."""
        if self.winfo_exists():  # Check if the widget still exists before accessing it
            for widget in self.winfo_children():
                widget.destroy()
        
        for fig in self.figures:
            plt.close(fig)
        
        self.destroy()