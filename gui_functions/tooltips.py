import tkinter as tk
from tkinter import ttk

def get_tooltips():
        # Tooltip dictionary
    tooltips = {
        "import_prices": """
            Select a .csv or .xlsx file. Accepted file formats:
            ◾ PERSEO daily PSV with forward curve.
            ◾ A file with daily spot prices followed by the forward curve. 
                ◾ The filename must contain the last date of spot prices with format 'YYYY-MM-DD'.
            ◾ A file with daily spot prices, must have two columns "Date", "Price". Forward data will be asked later.
            ◾ The "Date" column must have the format 'YYYY-MM-DD'.
        """,
        "import_forward": """
            Select a .csv or .xlsx file. Accepted file formats:
            ◾ A file with forward prices, must have two columns "Date", "Price".
                ◾ The first date must be one day after the last date in the spot prices dataset.
            ◾ The "Date" column must have the format 'YYYY-MM-DD'.
        """,
        "import_load": """ 
            "Select a .csv or .xlsx file with daily load data. Accepted file formats:
                ◾ A file with load values, must have two columns "Date", "Load".
                ◾ The "Date" column must have the format 'YYYY-MM-DD'.
        """,
        "clear_data": "Clear all imported data and reset the interface.",
        "calc_hedging": """ 
            "Insert hedgable periods trough forward, 0 is the first contract day.
                ◾ To hedge the first 30 days insert: '0, 30'.
                ◾ To hedge the first 90 days and the subsequent 90 days insert: '0, 90, 180' and so on.
        """,
    }

    return tooltips

def add_tooltip(self, widget, text):
    """Add a tooltip to a widget with a 2-second delay and neutral background."""
    tooltip = tk.Toplevel(widget, padx=5, pady=2)
    tooltip.wm_overrideredirect(True)
    tooltip.withdraw()

    # Use a neutral background color matching the default Tkinter theme
    tooltip.configure(bg=widget.winfo_toplevel().cget("bg"))

    tooltip_label = ttk.Label(tooltip, text=text, relief="solid", borderwidth=1)
    tooltip_label.pack()

    # Variable to track hover event
    self.tooltip_after_id = None

    def show_tooltip():
        """Display tooltip after delay."""
        tooltip.geometry(f"+{widget.winfo_rootx() + 10}+{widget.winfo_rooty() + 25}")
        tooltip.deiconify()

    def schedule_tooltip(event):
        """Schedule tooltip to appear after 800ms."""
        self.tooltip_after_id = widget.after(800, show_tooltip)

    def hide_tooltip(event):
        """Cancel scheduled tooltip if the cursor moves away."""
        if self.tooltip_after_id:
            widget.after_cancel(self.tooltip_after_id)
            self.tooltip_after_id = None
        tooltip.withdraw()

    widget.bind("<Enter>", schedule_tooltip)
    widget.bind("<Leave>", hide_tooltip)