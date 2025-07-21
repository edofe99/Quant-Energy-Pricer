import pandas as pd
import re
import os
from datetime import datetime
from tkinter import messagebox

def check_dataframe_date_column(df):
    if pd.to_datetime(df['Date'], errors='coerce').isnull().any():
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        if df['Date'].isnull().any():
            messagebox.showerror(
                "Error",
                "Invalid date format:\nExpecting a column 'Date' with one of the formats:\n"
                "- 'YYYY-MM-DD HH:mm'\n- 'YYYY-MM-DD'\n- 'DD/MM/YYYY'"
            )
            raise ValueError("Invalid date format in 'Date' column.")
    else:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    return df

def check_perseo_format(file_path):
    '''
    Checks if the name of the input file matches the PERSEO daily mail format
    Example pattern explanation:
    - Starts with "TRG - Curve Builder Gas - check@"
    - Followed by a date in YYYY_MM_DD format
    - Ends with time in HH_MM_SS and ".xlsx"
    
    If not a PERSEO format then extrapolate the date from the filename.
    '''
    # Extract the filename from the full path
    filename = os.path.basename(file_path)
    pattern = r"TRG - Curve Builder Gas - check@(\d{4}_\d{2}_\d{2})_\d{2}_\d{2}_\d{2}\.xlsx"
    
    # Check if the filename matches the pattern
    match = re.match(pattern, filename)
    if match:
        # Extract the date part (group 1 from regex)
        extracted_date_str = match.group(1)
        date = datetime.strptime(extracted_date_str, "%Y_%m_%d")
        return date, True
    else:
        try:
            match = re.search(r'\b\d{4}-\d{2}-\d{2}\b', filename)
            date = datetime.strptime(match.group(), '%Y-%m-%d')
            return date, False
        except (ValueError, AttributeError):
            # messagebox.showerror(
            #     "Error",
            #     "Incorrect filename:\n\n"
            #     "To extrapolate forward data, we need the current date.\n"
            #     "Please insert a date in the filename (format: 'YYYY-MM-DD').")
            return None, False

def import_spot_prices(file_path):
    '''
    Import the data from a xslx file or a csv file. The file must be in the PERSEO format or:
    - A csv/xslx/xls with 2 columns: 'Date', 'Close'.
    - The date column must contain valid dates in the format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:mm'.
    - After the day specified on the filename (format 'YYYY-MM-DD') there are forward prices.
    '''
    last_spot_date, perseo_format = check_perseo_format(file_path)

    if perseo_format:
        # Read the Excel file, skipping rows before the data starts
        df = pd.read_excel(file_path, header=None, skiprows=4)

        # Assign column names manually based on the structure
        df.columns = ['Data Inizio', 'IT_PSV_MTM', 'IT_PSV_ASK_MTM']

        # Drop extra rows that are not part of the data (e.g., merged cell titles in Row 6)
        df = df.iloc[2:].reset_index(drop=True)  # Start from Row 7 (after skipping)

        # Convert 'Data Inizio' to datetime format (if applicable)
        df['Data Inizio'] = pd.to_datetime(df['Data Inizio'], errors='coerce')

        df.rename(columns={'Data Inizio': 'Date', 'IT_PSV_MTM': 'Close'}, inplace=True)
        df.drop(columns=['IT_PSV_ASK_MTM'], inplace=True)

    else:
        # Determine file type based on the extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)

        # Check for 'Date' and 'Close' columns
        missing_columns = [col for col in ['Date', 'Price'] if col not in df.columns]
        if missing_columns:
            messagebox.showerror("Error", "Bad format:\nExpecting an Excel/CSV file with two columns: 'Date' and 'Price'.")
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
        else:
            df = check_dataframe_date_column(df)

    
    df = df.dropna()
    if last_spot_date is not None:
        spot_prices = df[df['Date'] <= last_spot_date].copy()  # Data on or before the date
        forward_prices = df[df['Date'] > last_spot_date].copy()   # Data after the date
    
        if spot_prices.isnull().values.any() or spot_prices.empty:
            messagebox.showerror("Error", "Invalid spot prices:\nThe given dataset of spot prices is empty or contains invalid values.")
            raise ValueError("Invalid date format in 'Date' column.")
        if forward_prices.isnull().values.any() or forward_prices.empty:
            messagebox.showerror("Error", "Invalid forward prices:\nThe given dataset of forward prices is empty or contains invalid values.")
            raise ValueError("Invalid date format in 'Date' column.")
        del df
        return spot_prices, forward_prices
    else:
        spot_prices = df.copy()
        if spot_prices.isnull().values.any() or spot_prices.empty:
            messagebox.showerror("Error", "Invalid spot prices:\nThe given dataset of spot prices is empty or contains invalid values.")
            raise ValueError("Invalid date format in 'Date' column.")
        del df
        return spot_prices, None

def import_forward_prices(file_path, last_spot_date):
    '''
    Import forward prices.
    - The first forward price must be 1 day after the last spot price.
    - The file should have a column "Date" with format "YYYY-MM-DD" and a column "Price".
    '''
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)

    # Check for 'Date' and 'Close' columns
    missing_columns = [col for col in ['Date', 'Price'] if col not in df.columns]
    if missing_columns:
        messagebox.showerror("Error", "Bad format:\nExpecting an Excel/CSV file with two columns: 'Date' and 'Price'.")
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
    else:
        df = check_dataframe_date_column(df)

    
    one_day_after = last_spot_date + pd.offsets.BDay(1)
    if one_day_after == df['Date'].iloc[0]:
        return df
    else:
        messagebox.showerror("Error", f"Incorrect start date:\nThe fist date in the dataset should be {one_day_after}.")
        raise ValueError(f"Incorrect start date: {one_day_after}")



def import_load(file_path):
    '''
    Import the data from a xslx file or a csv file. The file must contain:
    - A column called 'Date' with format "YYYY-MM-DD".
    - A column called 'Value' with format "YYYY-MM-DD".
    '''

     # Determine file type based on the extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)

    df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Value'}, inplace=True)
    df = check_dataframe_date_column(df)
    df = df.iloc[:, :2]  # Keep only the first two columns

    return df


