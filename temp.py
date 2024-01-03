import pandas as pd
from tradingview_ta import TA_Handler, Interval
import datetime
import os
import time

def check_interval():
    current_time = datetime.datetime.now().time()
    
    # Check if minutes component is a multiple of 5
    if (current_time.minute+1 % 5==0):
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

df = pd.read_csv("Symbols200.csv")
symbols = df["Symbol.NS"]

data = {}
for symbol in symbols:
    data[symbol] = []


while(True):
    if(check_interval()):
        for symbol in symbols:
            try:
                output = TA_Handler(
                                symbol=symbol[:len(symbol)-3],
                                screener="India",
                                exchange="NSE",
                                interval=Interval.INTERVAL_5_MINUTES
                            )

                indicators = output.get_indicators(["close", "EMA5", "VWAP"])
                close=indicators["close"]
                ema=indicators["EMA5"]
                vwap=indicators["VWAP"]
                data[symbol].append({"close":close, "ema":ema, "vwap":vwap})
                print(f"Added data for {symbol}")

            except Exception as e:
                print(f"{symbol}: {e}")

    elif(check_entry()):
        for symbol in symbols:
            try:
                output = TA_Handler(
                                symbol=symbol[:len(symbol)-3],
                                screener="India",
                                exchange="NSE",
                                interval=Interval.INTERVAL_5_MINUTES
                            )

                indicators = output.get_indicators(["close", "EMA5", "VWAP"])
                close=indicators["close"]
                ema=indicators["EMA5"]
                vwap=indicators["VWAP"]
                if(len(data[symbol])>2):
                    prevClose = data[symbol][-2]["close"]
                    prevEma = data[symbol][-2]["ema"]
                    prevVwap = data[symbol][-2]["vwap"]
                    if(prevEma<prevVwap and ema>vwap):
                        print(f"{symbol}: close={close}, ema={ema}, vwap={vwap}")
            except Exception as e:
                print(f"{symbol}: {e}")

    else:
        current_time = datetime.datetime.now().time()
        print(current_time)
        time.sleep(2)
        os.system('cls')