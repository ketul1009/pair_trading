import pandas as pd
from tradingview_ta import TA_Handler, Interval
import datetime
import os
import time

def check_interval():
    current_time = datetime.datetime.now().time()
    
    # Check if minutes component is a multiple of 5
    if ((current_time.minute+1) % 5==0) and(current_time.second>30):
        return True
    else:
        return False

def check_entry():
    current_time = datetime.datetime.now().time()
    
    # Check if minutes component is a multiple of 5
    if (current_time.minute) % 5 == 0:
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

df = pd.read_csv("Symbols50.csv")
symbols = df["Symbol"]

data = {}
for symbol in symbols:
    data[symbol] = []

while(True):
    if(check_interval()):
        for symbol in symbols:
            try:
                output = TA_Handler(
                                symbol=symbol,
                                screener="India",
                                exchange="NSE",
                                interval=Interval.INTERVAL_5_MINUTES
                            )

                indicators = output.get_indicators(["close", "EMA5", "VWAP"])
                close=indicators["close"]
                ema=indicators["EMA5"]
                vwap=indicators["VWAP"]
                data[symbol].append({"close":close, "ema":ema, "vwap":vwap})
                print(f"Adding...........")

            except Exception as e:
                print(f"{symbol}: {e}")

    elif(check_entry()):
        signals=[]
        for symbol in symbols:
            try:
                output = TA_Handler(
                                symbol=symbol,
                                screener="India",
                                exchange="NSE",
                                interval=Interval.INTERVAL_5_MINUTES
                            )

                indicators = output.get_indicators(["close", "EMA5", "VWAP"])
                close=indicators["close"]
                ema=indicators["EMA5"]
                vwap=indicators["VWAP"]
                print(f"Scanning...........")
                if(len(data[symbol])>2):
                    prevClose = data[symbol][-2]["close"]
                    prevEma = data[symbol][-2]["ema"]
                    prevVwap = data[symbol][-2]["vwap"]
                    if(prevEma<prevVwap and ema>vwap):
                        print(f"{symbol}: close={close}, ema={ema}, vwap={vwap}")
                        signals.append({"Time":datetime.datetime.now().time(),"Symbol":symbol, "Close":close, "EMA":ema, "VWAP":vwap})

            except Exception as e:
                print(f"{symbol}: {e}")

        update_csv_file("Signals.csv", signals)

    else:
        current_time = datetime.datetime.now().time()
        print(current_time)
        time.sleep(2)
        os.system('cls')

