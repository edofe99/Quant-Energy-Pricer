import tkinter as tk
from tkinter import ttk, filedialog, messagebox
# from tkcalendar import DateEntry
import model
import gui_functions
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class MainApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Quant Energy Pricer")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)  # Minimum size 800x600

        # Configure the main grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Bind closing event to ensure proper cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Create a notebook widget
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(sticky="nsew", padx=10, pady=10)

        # Configure the notebook grid to resize properly
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Create the Calculate tab
        self.calculate_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.calculate_frame, text="Calculate")
        self.calculate_frame.columnconfigure(0, weight=1)
        self.calculate_frame.rowconfigure(1, weight=1)  # For content resizing below the top section

        # Create the Results, Hedging, and Plots tabs
        self.results_frame = ttk.Frame(self.notebook)
        
        self.hedging_frame = ttk.Frame(self.notebook)
        self.hedging_frame.columnconfigure(0, weight=1)
        self.hedging_frame.rowconfigure(1, weight=1)  # For content resizing below the top section
        
        self.plots_frame = ttk.Frame(self.notebook)

        # Hide these tabs initially
        self.hidden_tabs = {
            "Results": self.results_frame,
            "Hedging": self.hedging_frame,
            "Plots": self.plots_frame,
        }

        # Import resources
        self.tooltips = gui_functions.get_tooltips()

        # Track the status of imports
        self.import_prices_done = False
        self.import_load_done = False

        # Add labeled sections to the Calculate tab
        self.add_import_data_section(self.calculate_frame)
        self.add_parameters_section(self.calculate_frame)
        self.add_buttons_section(self.calculate_frame)

    # Add the generate and clear data buttons
    def add_buttons_section(self, parent):
        """Add buttons below the 'Parameters' section."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, pady=10, sticky="ew")

        # Configure the button frame to make sure buttons align properly
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        # Create "Generate" button
        self.generate_button = ttk.Button(button_frame, text="Generate", state="disabled", command=self.on_generate_click)
        self.generate_button.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        # Create "Clear Data" button
        self.clear_data_button = ttk.Button(button_frame, text="Clear Data", command=self.on_clear_data_click)
        self.clear_data_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    # ---------------------------------------------------------------------------- #
    #                               Labeled sections                               #
    # ---------------------------------------------------------------------------- #

    def add_import_data_section(self, parent):
        """Create the 'Import Data' section."""
        self.import_data_frame = ttk.LabelFrame(parent, text="Import Data", padding=(10, 10))
        self.import_data_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # # Configure the frame for grid layout
        # self.import_data_frame.columnconfigure(2, weight=1)  # Adjust for Clear Data button alignment
        # self.import_data_frame.rowconfigure(3, weight=1)

        # Add "Import Prices/Forward Data" button with tooltip
        self.import_prices_button = ttk.Button(self.import_data_frame, text="Import Prices",
                                               command=self.on_import_prices_click)
        self.import_prices_button.grid(row=0, column=0, padx=5, pady=(5,15), sticky="w")

        # Status Label (Appears to the right of the button)
        self.import_prices_status = ttk.Label(self.import_data_frame, text="", foreground="green")
        self.import_prices_status.grid(row=0, column=1, padx=10, pady=(0,10), sticky="w")

        # Tooltip for Import Prices/Forward Data
        gui_functions.add_tooltip(self, self.import_prices_button, self.tooltips["import_prices"])

        # Add "Import Load" button with tooltip
        self.import_load_button = ttk.Button(self.import_data_frame, text="Import Load", command=self.on_import_load_click)
        self.import_load_button.grid(row=1, column=0, padx=5, pady=(10,5), sticky="w")

        # Status Label (Appears to the right of the button)
        self.import_load_status = ttk.Label(self.import_data_frame, text="", foreground="green")
        self.import_load_status.grid(row=1, column=1, padx=5, pady=(10,5), sticky="w")

        # Tooltip for Import Load
        gui_functions.add_tooltip(self, self.import_load_button, self.tooltips["import_load"])

    def add_parameters_section(self, parent):
        """Create the 'Parameters' section."""
        parameters_frame = ttk.LabelFrame(parent, text="Parameters", padding=(10, 10))
        parameters_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Configure grid for input layout
        for i in range(3):  # Adjust rows
            parameters_frame.rowconfigure(i, weight=1)
            parameters_frame.columnconfigure(i, weight=1)

        # ----------------------------- Date input fields ---------------------------- #
        
        # Contract Start
        ttk.Label(parameters_frame, text="Calibrate from").grid(row=0, column=0, padx=5, pady=5, sticky="n")
        self.calibration_start_picker = ttk.Entry(parameters_frame, state="disabled")
        self.calibration_start_picker.grid(row=0, column=0, padx=5, pady=30, sticky="n")
        self.calibration_start_picker.bind("<KeyRelease>", lambda e: self.validate_dates())
        self.calibration_start_picker.bind("<KeyRelease>", lambda e: self.check_all_validations())


        # Contract Start
        ttk.Label(parameters_frame, text="Contract Start").grid(row=0, column=1, padx=5, pady=5, sticky="n")
        self.contract_start_picker = ttk.Entry(parameters_frame, state="disabled")
        self.contract_start_picker.grid(row=0, column=1, padx=5, pady=30, sticky="n")
        self.contract_start_picker.bind("<KeyRelease>", lambda e: self.validate_dates())
        self.contract_start_picker.bind("<KeyRelease>", lambda e: self.check_all_validations())

        # Contract End
        ttk.Label(parameters_frame, text="Contract End").grid(row=0, column=2, padx=5, pady=5, sticky="n")
        self.contract_end_picker = ttk.Entry(parameters_frame, state="disabled")
        self.contract_end_picker.grid(row=0, column=2, padx=5, pady=30, sticky="n")
        self.contract_end_picker.bind("<KeyRelease>", lambda e: self.validate_dates())
        self.contract_end_picker.bind("<KeyRelease>", lambda e: self.check_all_validations())




        # ----------------------------- Other parameters ----------------------------- #

        # Simulations
        ttk.Label(parameters_frame, text="Simulations").grid(row=1, column=0, padx=5, pady=5, sticky="n")
        self.simulations_var = tk.StringVar(value="1000")  # Set default value
        self.simulations_entry = ttk.Entry(parameters_frame, state="disabled")  # Initially disabled
        self.simulations_entry.grid(row=1, column=0, padx=5, pady=30, sticky="n")
        # Check if valid entry
        self.simulations_entry.bind("<KeyRelease>", lambda e: self.validate_simulations())
        # Check if we can start calculations
        self.simulations_entry.bind("<KeyRelease>", lambda e: self.check_all_validations())

        # Risk-Free Rate
        ttk.Label(parameters_frame, text="Risk-Free Rate (%)").grid(row=1, column=1, padx=5, pady=5, sticky="n")
        self.risk_free_rate_entry = ttk.Entry(parameters_frame, state="disabled")  # Initially disabled
        self.risk_free_rate_entry.grid(row=1, column=1, padx=5, pady=30, sticky="n")
        # Check if valid entry
        self.risk_free_rate_entry.bind("<KeyRelease>", lambda e: self.validate_risk_free_rate())
        # Check if we can start calculations
        self.risk_free_rate_entry.bind("<KeyRelease>", lambda e: self.check_all_validations())

        # Hurdle rate (RAROC)
        ttk.Label(parameters_frame, text="Hurdle Rate (%)").grid(row=1, column=2, padx=5, pady=5, sticky="n")
        self.raroc_entry = ttk.Entry(parameters_frame, state="disabled")  # Initially disabled
        self.raroc_entry.grid(row=1, column=2, padx=5, pady=30, sticky="n")
        # Check if valid entry
        self.raroc_entry.bind("<KeyRelease>", lambda e: self.validate_raroc())
        # Check if we can start calculations
        self.raroc_entry.bind("<KeyRelease>", lambda e: self.check_all_validations())

        #TODO delete everything on genearte

        # Store the frame and widgets for enabling/disabling later
        self.parameters_widgets = [
            self.calibration_start_picker,
            self.contract_start_picker,
            self.contract_end_picker,
            self.simulations_entry,
            self.risk_free_rate_entry,
            self.raroc_entry
        ]


    def add_results_frame(self):
        """Adds four text labels in a 2x2 grid layout on the Results tab."""
        # Clear previous content in the results frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Settings frame
        results_text_frame = ttk.LabelFrame(self.results_frame, text="Metrics", padding=(10, 10))
        results_text_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=(10,0), sticky="nsew")

        # Ensure settings expand properly
        for i in range(2):  # 2 columns instead of 3
            results_text_frame.columnconfigure(i, weight=1)

        # Delete previous chart if exists
        if hasattr(self, 'volume_risk_chart') and self.volume_risk_chart:
            self.volume_risk_chart.close_navigator()

        # Define text strings
        text_strings = [
            f"Contract Price: {self.mycontract.optimalPrice:.2f}€",
            f"Volume Risk: {self.mycontract.volume_risk_var:.2f}€",
            f"Final Contract Price: {self.mycontract.volume_risk_var+self.mycontract.optimalPrice:.2f}€",
            # f"Hurdle Rate: {self.mycontract.hurdleRate*100:.0f}%",
            # f"VaR(95%): {self.mycontract.var:.3f}€",
        ]

        # Display the text strings in a 2x2 grid layout
        for i, text in enumerate(text_strings):
            row = i // 3
            col = i % 3
            label = ttk.Label(results_text_frame, text=text)
            label.grid(row=row, column=col, padx=15, pady=10, sticky="nsew")

        # Make figure smaller
        volume_risk_fig = self.mycontract.volume_risk_chart
        volume_risk_fig.set_size_inches(6, 4)

        # Charts frame
        self.volume_risk_chart_frame = ttk.LabelFrame(self.results_frame, text="Chart", padding=(10, 0))
        self.volume_risk_chart_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(0,5), sticky="nsew")

        # Configure grid weights
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(1, weight=1)
        self.volume_risk_chart_frame.columnconfigure(0, weight=1)
        self.volume_risk_chart_frame.rowconfigure(0, weight=1)

        # Add chart navigation
        self.volume_risk_chart = gui_functions.ChartNavigator(self.volume_risk_chart_frame, volume_risk_fig)
        self.volume_risk_chart.pack(expand=True, fill=tk.BOTH)

    def add_hedging_settings_section(self):

        hedging_settings_frame = ttk.LabelFrame(self.hedging_frame, text="Settings", padding=(10, 10))
        hedging_settings_frame.grid(row=0, column=0, padx=10, pady=(10,0), sticky="nsew")

        # Configure grid for input layout
        for i in range(3):  # Adjust rows
            hedging_settings_frame.rowconfigure(i, weight=1)
            hedging_settings_frame.columnconfigure(i, weight=1)

        ttk.Label(hedging_settings_frame, text="Hedging contracts").grid(row=0, column=0, padx=5, pady=5, sticky="n")
        self.hedging_contracts = ttk.Entry(hedging_settings_frame)
        self.hedging_contracts.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.hedging_contracts.bind("<KeyRelease>", lambda e: self.validate_hedging_contracts())

        self.calc_hedging_button = ttk.Button(hedging_settings_frame, text="Calculate Hedging", state="disabled", command=self.on_calc_hedging_click)
        self.calc_hedging_button.grid(row=0, column=2, padx=5, pady=5)
        gui_functions.add_tooltip(self, self.calc_hedging_button, self.tooltips["calc_hedging"])
        
        # --------------------------- Hedging plot section --------------------------- #

        self.hedging_plot_frame = ttk.LabelFrame(self.hedging_frame, text="Charts", padding=(10, 0))
        self.hedging_plot_frame.grid(row=1, column=0, padx=10, pady=(0,5), sticky="nsew")

    # ---------------------------------------------------------------------------- #
    #                           On button press functions                          #
    # ---------------------------------------------------------------------------- #

    def on_import_prices_click(self):
        """Handle click on 'Import Prices/Forward Data' button."""
        spot_file_path = filedialog.askopenfilename(title="Select a file containg spot and forward prices", 
                                      filetypes=[
                                                        ("Excel/CSV files (.xlsx, .xls, .csv)", "*.xlsx"),
                                                        ("Excel/CSV files (.xlsx, .xls, .csv)", "*.xls"),
                                                        ("Excel/CSV files (.xlsx, .xls, .csv)", "*.csv"),
                                                        ("All files", "*.*")
                                                    ])
        
        self.spot_data, self.forward_data = gui_functions.import_spot_prices(spot_file_path)

        if self.forward_data is None:
            self.import_prices_status.config(foreground="orange")
            self.import_prices_status.config(text="Could not find forward data.\nOnly spot data was imported.")
            # Create a button to import forward prices
            self.import_forward_button = ttk.Button(self.import_data_frame, text="Import Forward Data",
                                               command=self.on_import_forward_click)
            self.import_forward_button.grid(row=0, column=2, padx=5, pady=(5,15), sticky="w")
            # Tooltip for Import Prices/Forward Data
            gui_functions.add_tooltip(self, self.import_forward_button, self.tooltips["import_forward"])
        else:
            self.import_prices_done = True
            self.import_prices_status.config(foreground="green")
            self.import_prices_status.config(text="Data Imported")
            self.update_parameters_buttons_state()

    def on_import_forward_click(self):
        """Handle click on 'Forward Data' button."""
        forward_file_path = filedialog.askopenfilename(title="Select a file containg forward prices", 
                                      filetypes=[
                                                        ("Excel/CSV files (.xlsx, .xls, .csv)", "*.xlsx"),
                                                        ("Excel/CSV files (.xlsx, .xls, .csv)", "*.xls"),
                                                        ("Excel/CSV files (.xlsx, .xls, .csv)", "*.csv"),
                                                        ("All files", "*.*")
                                                    ])


        self.forward_data = gui_functions.import_forward_prices(forward_file_path, self.spot_data.iloc[-1,0])

        self.import_forward_button.destroy()
        self.import_prices_done = True
        self.import_prices_status.config(foreground="green")
        self.import_prices_status.config(text="Data Imported")
        self.update_parameters_buttons_state()

    def on_import_load_click(self):
        """Handle click on 'Import Load' button."""

        load_file_path = filedialog.askopenfilename(title="Select a file containg load",
                                                        filetypes=[
                                                        ("Excel/CSV files (.xlsx, .xls, .csv)", "*.xlsx"),
                                                        ("Excel/CSV files (.xlsx, .xls, .csv)", "*.xls"),
                                                        ("Excel/CSV files (.xlsx, .xls, .csv)", "*.csv"),
                                                        ("All files", "*.*")
                                                    ])
        
        self.load_data = gui_functions.import_load(load_file_path)
        self.import_load_done = True
        self.import_load_status.config(text="Load Imported")
        self.update_parameters_buttons_state()

    def on_clear_data_click(self, only_charts = False):
        """Clear data and reset the import status."""

        # Delete results volume risk chart
        if hasattr(self, 'volume_risk_chart') and self.volume_risk_chart:
            self.volume_risk_chart.close_navigator()
        # Delete output charts
        if hasattr(self, 'result_charts') and self.result_charts:
            self.result_charts.close_navigator()
        # Delete hedging charts
        if hasattr(self, 'hedging_charts') and self.hedging_charts:
            self.hedging_charts.close_navigator()

        # Close notebook tabs
        for tab_name, frame in self.hidden_tabs.items():
            if str(frame) in self.notebook.tabs():  # Convert the frame to its string identifier
                self.notebook.forget(frame)
        
        # Remove forward button if it was generated
        if hasattr(self, 'import_forward_button'):
            self.import_forward_button.destroy()
            del self.import_forward_button  # Remove the attribute reference

        if not only_charts:
            self.import_prices_done = False
            self.import_load_done = False
            self.import_prices_status.config(text="")
            self.import_load_status.config(text="")
            self.update_parameters_buttons_state()  # Reset parameter widgets and Generate button

    def on_generate_click(self):
        
        # Close charts if present
        self.on_clear_data_click(only_charts=True)

        # -------------------------- Execute the calibration ------------------------- #

        self.spot_data = self.spot_data[self.spot_data['Date'] >= self.calibration_start_picker.get()].reset_index(drop=True)
        self.load_data_full = self.load_data.copy()
        self.load_data = self.load_data[self.load_data['Date'] >= self.calibration_start_picker.get()].reset_index(drop=True)
        self.forward_data.index = range(self.spot_data.index[-1] + 1, self.spot_data.index[-1] + 1 + len(self.forward_data))
        
        risk_free = float(self.risk_free_rate_entry.get()) / 100 / 365
        raroc_hurdle_rate = float(self.raroc_entry.get()) / 100
        figures = []

        self.mycontract = model.Contract(
            self.spot_data, self.forward_data, (self.load_data,self.load_data_full) ,
            int(self.simulations_entry.get()), 
            risk_free,
            self.contract_start_picker.get(),
            self.contract_end_picker.get(),
            30, raroc_hurdle_rate,
            figures
        )

        self.mycontract.contractCalculation()

        # ------------------------------- Show results ------------------------------- #

        # Add the ChartNavigator widget inside the "Plots" tab
        self.result_charts = gui_functions.ChartNavigator(self.plots_frame, figures)
        self.result_charts.pack(expand=True, fill=tk.BOTH)

        # Show hidden tabs
        for tab_name, frame in self.hidden_tabs.items():
            if frame not in self.notebook.tabs():
                self.notebook.add(frame, text=tab_name)
        
        # hedging
        self.add_hedging_settings_section()

        # Add information text to the Results frame
        self.add_results_frame()


    def on_calc_hedging_click(self):
        
        # Check if charts already exist
        if hasattr(self, 'hedging_charts') and self.hedging_charts:
            self.hedging_charts.close_navigator()

        # for widget in self.hedging_plot_frame.winfo_children():
        #     widget.destroy()

        hedging_strategy = model.Hedging(self.mycontract)
        eta = hedging_strategy.calc_eta(self.hedging_periods)

        hedging_strategy.hedging(eta)

        # Add the ChartNavigator widget inside the "Plots" tab
        if len(hedging_strategy.figures) >= 1:
            for fig in hedging_strategy.figures:
                fig.set_size_inches(6, 4)  # Adjust width & height to fit within the window
            self.hedging_charts = gui_functions.ChartNavigator(self.hedging_plot_frame, hedging_strategy.figures)
            self.hedging_charts.pack(expand=True, fill=tk.BOTH)

    # ---------------------------------------------------------------------------- #
    #                               Helper functions                               #
    # ---------------------------------------------------------------------------- #

    def validate_dates(self):
        """Enable 'Generate' button only when all fields are valid."""
        try:
            calibration_start = self.calibration_start_picker.get() 
            calibration_start_date = datetime.strptime(calibration_start, "%Y-%m-%d")
            
            # self.contract_start_picker.configure(mindate=calibration_start_date)
            contract_start = self.contract_start_picker.get()
            contract_start_date = datetime.strptime(contract_start, "%Y-%m-%d")
            
            # self.contract_end_picker.configure(mindate=contract_start_date)
            contract_end = self.contract_end_picker.get()
            contract_end_date = datetime.strptime(contract_end, "%Y-%m-%d")

            if (contract_end_date <= contract_start_date or
                calibration_start_date >= contract_start_date or
                calibration_start_date <=  self.data_start or
                not (contract_start_date >= self.min_date and contract_start_date <= self.max_date) or
                not (contract_end_date >= self.min_date and contract_end_date <= self.max_date) ):
                return False
            else:
                return True
        except ValueError:
            return False
    
    def validate_simulations(self):
        # Validate that the simulations input is an integer greater than or equal to 1.
        value = self.simulations_entry.get()
        if not value.isdigit() or int(value) < 1:
            self.simulations_entry.config(foreground="red")
            self.generate_button.config(state="disabled")
            return False
        else:
            self.simulations_entry.config(foreground="black")
            return True

    def validate_risk_free_rate(self):
        """Validate that the risk-free rate input is a number with up to three decimal places."""
        value = self.risk_free_rate_entry.get()
        if value.replace(".", "", 5).isdigit() and float(value) >= 0:
            self.risk_free_rate_entry.config(foreground="black")
            return True
        else:
            self.risk_free_rate_entry.config(foreground="red")
            return False
    
    def validate_raroc(self):
        """Validate that the risk-free rate input is a number with up to three decimal places."""
        value = self.raroc_entry.get()
        if value.replace(".", "", 5).isdigit() and float(value) >= 0:
            self.raroc_entry.config(foreground="black")
            return True
        else:
            self.raroc_entry.config(foreground="red")
            return False
    
    def check_all_validations(self):
        """Check all validation functions and enable/disable the 'Generate' button."""
        all_valid = (
            self.validate_dates() and
            self.validate_simulations() and
            self.validate_risk_free_rate() and
            self.validate_raroc
        )

        if all_valid:
            self.generate_button.config(state="normal")  # Enable button
        else:
            self.generate_button.config(state="disabled")  # Disable button

    def update_parameters_buttons_state(self):
        """Enable or disable the parameter section widgets and Generate button based on the import status."""
        if self.import_prices_done and self.import_load_done:
            
            # ------------------------- Chek start and end dates ------------------------- # 
            self.data_start = max(self.spot_data['Date'].iloc[0], self.load_data['Date'].iloc[0])
            # last_date = min(self.spot_data['Date'].iloc[-1], self.load_data['Date'].iloc[-1])

            # Filter both DataFrames to start at the same date
            self.spot_data = self.spot_data[self.spot_data['Date'] >=  self.data_start]
            self.load_data = self.load_data[self.load_data['Date'] >=  self.data_start]
            
            # Check if both dataframes ends at the same date
            if self.spot_data['Date'].iloc[-1] != self.load_data['Date'].iloc[-1]:
                messagebox.showerror("Date error", f"""Spot prices ends on {self.spot_data['Date'].iloc[-1]} while load ends on {self.load_data['Date'].iloc[-1]}\n
                                     The end date must be the same.""")
                # Delete previously imported data
                self.on_clear_data_click()
                raise ValueError(f"Incorrect end dates: {self.spot_data['Date'].iloc[-1]}, {self.load_data['Date'].iloc[-1]}")

            # Reset index
            self.spot_data = self.spot_data.reset_index(drop=True)
            self.load_data = self.load_data.reset_index(drop=True)

            # Enable all widgets in the parameters section
            for widget in self.parameters_widgets:
                widget.config(state="normal")
            
            self.simulations_entry.insert(0, "1000")  # Insert default value directly
            self.raroc_entry.insert(0, "8")  # Insert default value directly

            last_spot_day = self.spot_data['Date'].iloc[-1]
            self.min_date = last_spot_day + timedelta(days=1)
            self.max_date = self.forward_data['Date'].iloc[-1]
            # Default use last yeaar for calibrating
            start_of_last_year = pd.Timestamp(year=last_spot_day.year - 1, month=1, day=1)
            start_of_next_year = pd.Timestamp(year=last_spot_day.year + 1, month=1, day=1)
            end_of_next_year = pd.Timestamp(year=last_spot_day.year + 1, month=12, day=31)

            # self.calibration_start_picker.set_date(start_of_last_year)
            self.calibration_start_picker.insert(0, start_of_last_year.strftime('%Y-%m-%d'))
            # self.calibration_start_picker.configure(mindate=self.spot_data['Date'].iloc[0],
            #                                     maxdate=self.min_date)
            # self.contract_start_picker.configure(mindate=self.min_date,
            #                                     maxdate=self.max_date)    
            # self.contract_end_picker.configure(mindate=self.min_date,
            #                                     maxdate=self.max_date)
            
            # Set default date picker values
            # self.contract_start_picker.set_date(start_of_next_year)
            # self.contract_end_picker.set_date(end_of_next_year)
            self.contract_start_picker.insert(0,start_of_next_year.strftime('%Y-%m-%d'))
            self.contract_end_picker.insert(0,end_of_next_year.strftime('%Y-%m-%d'))

        else:
            # Disable all widgets in the parameters section
            for widget in self.parameters_widgets:
                if isinstance(widget, ttk.Entry):  # If it's a text field, clear it
                    widget.delete(0, tk.END)
                # elif isinstance(widget, DateEntry):  # If it's a DateEntry, reset it
                #     widget.set_date("")
                widget.config(state="disabled")  # Disable after clearing
            #self.generate_button.config(state="disabled")  # Disable Generate button

    def validate_hedging_contracts(self):
        # Step 1: Split and strip spaces
        elements = self.hedging_contracts.get().split(",")
        elements = [e.strip() for e in self.hedging_contracts.get().split(",")]
        num_list = []

        # Step 2: Convert to integers and validate numbers
        for item in elements:
            if not item.isdigit():  # Check if item is a number
                self.calc_hedging_button.config(state="disabled")
                return
            num_list.append(int(item))

        # Step 3: Check that the list is not empty
        if len(num_list) == 0:
            self.calc_hedging_button.config(state="disabled")
            return

        # Step 4: Check that the numbers are strictly increasing
        for i in range(len(num_list) - 1):
            if num_list[i] >= num_list[i + 1]:  # Check for increasing order
                self.calc_hedging_button.config(state="disabled")
                return "Error: Numbers are not strictly increasing."

        self.hedging_periods = num_list

        # If every check pass, enable the button        
        self.calc_hedging_button.config(state="normal")


    # ---------------------------------------------------------------------------- #
    #                        Properly close the application                        #
    # ---------------------------------------------------------------------------- #
    
    def delete_charts(self, only_hedging = False):
        """ Ensure Matplotlib figures are closed and properly terminate the script. """
        if not only_hedging:
            if hasattr(self, 'result_charts') and self.result_charts:
                for fig in self.result_charts.figures:
                    plt.close(fig)  # Close all figures

        if hasattr(self, 'hedging_charts') and self.hedging_charts:
            for fig in self.hedging_charts.figures:
                plt.close(fig)  # Close all figures



    def on_close(self):

        self.root.destroy()  # Close Tkinter window
        print("Application closed. Exiting...")
        self.root.quit()  # Ensure script exits completely

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    available_themes = style.theme_names() # return a list with available themes
    
    # Try to set aqua theme, if on MacOS, else use 'vista' theme if on Windows
    # If none of these available (you're on linux), use fallback theme 'clam'
    # Available fallback themes: 'clam', 'alt', 'default', 'classic'
    preferred_themes = ['aqua', 'vista', 'clam']

    for theme in preferred_themes:
        if theme in available_themes:
            style.theme_use(theme)
            break

    app = MainApp(root)
    root.mainloop()
