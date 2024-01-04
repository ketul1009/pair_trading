import utility_functions as uf
import pandas as pd
import datetime
import os
import time

def check_interval():
    current_time = datetime.datetime.now().time()
    
    # Check if minutes component is a multiple of 5
    if (current_time.minute+4 % 5==0) and(current_time.second>30):
        return True
    else:
        return False

def update_csv_file(csv_file_path, new_data):
    # Read the existing CSV file
    try:
        df = pd.read_csv(csv_file_path)

    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        df = pd.DataFrame()

    # Create a DataFrame for the new data
    new_data_df = pd.DataFrame(new_data)

    # Concatenate the existing DataFrame and the new data DataFrame
    df = pd.concat([df, new_data_df], ignore_index=True)

    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)

signals = [
    {
        "Time":datetime.datetime.now().time(),
        "Symbol":"SBIN",
        "Close":100,
        "EMA":101,
        "VWAP":100.5
    },
    {
        "Time":datetime.datetime.now().time(),
        "Symbol":"SBIN",
        "Close":100,
        "EMA":101,
        "VWAP":100.5
    },
]

update_csv_file("Signals.csv", signals)