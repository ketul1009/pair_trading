import utility_functions as uf
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import time

datalocation = "Historical Data"
pair = ["", "SBIN", "ICICIBANK"]
start = "2023-04-03"
end = "2024-01-23"
lookback = 30
time="Date"

x = pd.read_csv(f"{datalocation}/{pair[1]}.csv")
y = pd.read_csv(f"{datalocation}/{pair[2]}.csv")

x = x[pd.to_datetime(x[time]).between(pd.to_datetime(start), pd.to_datetime(end))]
y = y[pd.to_datetime(y[time]).between(pd.to_datetime(start), pd.to_datetime(end))]

# x = x[(x['Date']>="2023-03-06") & (x['Date']<="2023-12-29")]
# y = y[(y['Date']>="2023-03-06") & (y['Date']<="2023-12-29")]

# x = x["Close"]
# y = y["Close"]

results1, ratio1, p_value1 = uf.linearRegression(y["Close"], x["Close"])
results2, ratio2, p_value2 = uf.linearRegression(x["Close"], y["Close"])

print(ratio1, ratio2, p_value1, p_value2)

print(uf.get_std_err(y, x))

"""
def backtest(lots, trades, intraday=False, n_bars=10, interval=Interval.in_daily, fut_contract=0, export=None):

    tv = TvDatafeed()
    pnl = []

    for index, trade in trades.iterrows():
        stockX = trade["Stock X"]
        stockY = trade["Stock Y"]
        df1 = tv.get_hist(symbol=stockX,exchange='NSE',interval=interval, n_bars=n_bars, fut_contract=fut_contract)
        df2 = tv.get_hist(symbol=stockY,exchange='NSE',interval=interval, n_bars=n_bars, fut_contract=fut_contract)
        if(intraday):
          df1.index = pd.to_datetime(df1.index) + pd.Timedelta(hours=5, minutes=30)
          df2.index = pd.to_datetime(df2.index) + pd.Timedelta(hours=5, minutes=30)
        else:
          dates=[]
          for date in df1.index:
            dates.append(str(date)[:11])
          df1.index = pd.to_datetime(dates)
          df2.index = pd.to_datetime(dates)

        entryDate = pd.to_datetime(trade["entry"])
        exitDate = pd.to_datetime(trade["exit"])
        type = trade["type"]
        lotX = get_lot_size(lots, stockX)
        lotY = get_lot_size(lots, stockY)
        if(type=="short"):
            buyX = df1.loc[exitDate]["close"]
            buyY = df2.loc[entryDate]["close"]
            sellX = df1.loc[entryDate]["close"]
            sellY = df2.loc[exitDate]["close"]
            unrealized_pnl = (sellX-buyX)*lotX + (sellY-buyY)*lotY
            realized_pnl = (sellX-buyX)*lotX + (sellY-buyY)*lotY - get_txn_cost(buyX, sellX, lotX) - get_txn_cost(buyY, sellY, lotY)
            charges = unrealized_pnl-realized_pnl
            pnl.append({"X":stockX, "Y":stockY, "Type": type,"entry":entryDate, "exit":exitDate, "Unrealized pnl":unrealized_pnl, "charges":charges, "realized_pnl":realized_pnl})

        elif(type=="long"):
            buyX = df1.loc[entryDate]["close"]
            buyY = df2.loc[exitDate]["close"]
            sellX = df1.loc[exitDate]["close"]
            sellY = df2.loc[entryDate]["close"]
            unrealized_pnl = (sellX-buyX)*lotX + (sellY-buyY)*lotY
            realized_pnl = (sellX-buyX)*lotX + (sellY-buyY)*lotY - get_txn_cost(buyX, sellX, lotX) - get_txn_cost(buyY, sellY, lotY)
            charges = unrealized_pnl-realized_pnl
            pnl.append({"X":stockX, "Y":stockY, "Type": type, "entry":entryDate, "exit":exitDate, "Unrealized pnl":unrealized_pnl, "charges":charges, "realized_pnl":realized_pnl})

    pnl_df = pd.DataFrame(pnl)
    if(export!=None):
        pnl_df.to_excel(f"{export}", index=False)
    return pnl_df
"""