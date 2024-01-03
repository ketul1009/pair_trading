import yfinance as yf
import os
import pandas as pd
import csv
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval

import Exceptions

def fetch_data(export, start=None, end=None, symbols=[], filename=None, interval ="1d"):

    if(len(symbols)==0 and filename==None):
        raise Exceptions.CustomException("No symbols list or symbol file provided")

    elif(filename!=None):
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for x in csvreader:
                try:
                    # Set the ticker
                    ticker = x[0]
                    
                    data = yf.download(f"{ticker}.NS", start=start, end=end, interval=interval)
                
                    #export data
                    path = f"{export}/{str(ticker)}.csv"
                    data.to_csv(path)
                except Exception as e:
                    print(f"Error occured for Symbol = {x}: {e}")

    else:
        for symbol in symbols:
            try:
                # Set the ticker
                ticker = symbol
                    
                data = yf.download(f"{ticker}.NS", start=start, end=end, interval=interval)
                
                #export data
                path = f"{export}/{str(ticker)}.csv"
                data.to_csv(path)
            except Exception as e:
                print(f"Error occured for Symbol = {symbol}: {e}")

def get_corr_matrix(folder_path, start, end, time="Date"):

    combined = pd.DataFrame()

    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.csv'):
            try:
                stock_name = os.path.splitext(file)[0]
                stock_name=stock_name[2:len(stock_name)-5]
                file_path = os.path.join(folder_path, file)
                data = pd.read_csv(file_path)
                data = data[pd.to_datetime(data[time]).between(pd.to_datetime(start), pd.to_datetime(end))]
                data = data["Close"].pct_change()
                combined[stock_name]=data.values
            except Exception as e:
                print(stock_name, e)

    return combined.corr()

def get_pairs(corr_matrix, threshold, export=None):

    pairs=[]
    for i in range(corr_matrix.shape[0]):  # Iterate over rows
        for j in range(corr_matrix.shape[1]):  # Iterate over columns
            element = corr_matrix.iloc[i, j]
            if(element>threshold and element<1):
                if [corr_matrix.columns[j], corr_matrix.index[i], element] not in pairs:
                    pairs.append([corr_matrix.index[i], corr_matrix.columns[j], element])

    pairs = pd.DataFrame(pairs, columns=["Stock A", "Stock B", "Correlation"])

    if(export!=None):
        pairs.to_csv(f"{export}", index=False)

    return pairs

def get_datasheet(df1, df2, start, end, time="Date", export=None):

    df1 = df1[pd.to_datetime(df1[time]).between(pd.to_datetime(start), pd.to_datetime(end))]
    new_df1=pd.DataFrame()
    new_df1["Close"] = df1["Close"]
    new_df1["Pct Change"] = df1["Close"].pct_change()
    new_df1["Absolute"] = df1["Close"].diff()

    df2 = df2[pd.to_datetime(df1[time]).between(pd.to_datetime(start), pd.to_datetime(end))]
    new_df2=pd.DataFrame()
    new_df2["Close"] = df2["Close"]
    new_df2["Pct Change"] = df2["Close"].pct_change()
    new_df2["Absolute"] = df2["Close"].diff()

    data_sheet=pd.DataFrame()
    data_sheet[time]=df1[time]
    data_sheet["Spread"]=new_df1["Absolute"].values-new_df2["Absolute"].values
    data_sheet["Differential"]=new_df1["Close"].values-new_df2["Close"].values
    data_sheet["Ratio"]=new_df1["Close"].values/new_df2["Close"].values

    mean = data_sheet["Ratio"].mean()
    std_dev = data_sheet["Ratio"].std()

    density_curve = norm.cdf(data_sheet["Ratio"], loc=mean, scale=std_dev)
    data_sheet["Density Curve"]=density_curve
    data_sheet["Mean"] = [mean for i in range(len(data_sheet))]
    
    if(export!=None):
        data_sheet.to_csv(f"{export}", index=False)

    return data_sheet

def get_signals(data_sheet, start, end, time="Date", export=None):

    signals = []
    trade={}
    openTrade=False
    tradeType=""
    for index, row in data_sheet.iterrows():
        if(pd.to_datetime(row[time])>pd.to_datetime(start) and pd.to_datetime(row[time])<pd.to_datetime(end)):
            if(row["Density Curve"]>0.975 and row["Density Curve"]<0.997 and not openTrade):
                trade["type"]="short"
                trade["entry"]=row[time]
                openTrade=True
                tradeType="s"

            elif(row["Density Curve"]<0.025 and row["Density Curve"]>0.003 and openTrade):
                trade["type"]="long"
                trade["entry"]=row[time]
                openTrade=True
                tradeType="l"

            elif(openTrade and tradeType=="s"):
                if(row["Density Curve"]<0.975 or row["Density Curve"]>0.997):
                    trade["exit"]=(row[time])
                    openTrade=False
                    tradeType=""
                    signals.append(trade)
                    trade={}
            
            elif(openTrade and tradeType=="l"):
                if(row["Density Curve"]>0.025 or row["Density Curve"]<0.003):
                    trade["exit"]=(row[time])
                    openTrade=False
                    tradeType=""
                    signals.append(trade)
                    trade={}

    if(export!=None):
        signals = pd.DataFrame(signals, columns=["Type", "Entry", "Exit"])
        signals.to_csv(f"{export}", index=False)

    return signals

def get_trades(pairs, dataLocation, start, end, time="Date", export=None):
    trades=[]
    for index, row in pairs.iterrows():
        stockA=row["Stock A"]
        stockB=row["Stock B"]
        pairName=stockA+"/"+stockB
        x = pd.read_csv(f"{dataLocation}/{stockA}.csv")
        y = pd.read_csv(f"{dataLocation}/{stockB}.csv")
        datasheet = get_datasheet(x, y, start, end)
        signals = get_signals(datasheet, start, end)
        if(len(signals)>0):
            for signal in signals:
                trades.append([pairName, signal["type"], signal["entry"], signal["exit"]])

    trades = pd.DataFrame(trades, columns=["Pair", "Type", "Entry", "Exit"])
    if(export!=None):
        trades.to_csv(f"{export}", index=False)

    return trades

def linearRegression(df1, df2):

    # Add a constant term to the independent variable
    df2_with_const = sm.add_constant(df2)

    # Ensure that both dataframes have the same index
    df1.index = df2_with_const.index

    # Fit the linear regression model
    model = sm.OLS(df1, df2_with_const).fit()

    residuals = model.resid

    # Standard deviation of residuals
    std_dev_residuals = residuals.std()

    # Access Standard Error of Intercept
    standard_error_intercept = model.bse['const']

    errorRatio = standard_error_intercept/std_dev_residuals

    result = adfuller(residuals)

    # Extracting test statistics and p-value
    adf_statistic = result[0]
    p_value = result[1]

    return model.summary(), errorRatio, p_value

def get_LR_pairs(symbols, dataLocation, start, end, time="Date", export=None):

    pairs = []
    for row1 in symbols.iterrows():
        for row2 in symbols.iterrows():
            industry1 = row1[1][1]
            industry2 = row2[1][1]
            symbol1 = row1[1][2]
            symbol2 = row2[1][2]

            if(industry1 == industry2 and symbol1 != symbol2 and [symbol2, symbol1] not in pairs):
                try:
                    df1 = pd.read_csv(f"{dataLocation}/{symbol1}.csv")
                    df2 = pd.read_csv(f"{dataLocation}/{symbol2}.csv")

                    # Filter data based on the date
                    df1 = df1[pd.to_datetime(df1[time]).between(pd.to_datetime(start), pd.to_datetime(end))]
                    df2 = df2[pd.to_datetime(df2[time]).between(pd.to_datetime(start), pd.to_datetime(end))]

                    # Extract the "Close" column
                    df1 = df1["Close"]
                    df2 = df2["Close"]

                    results1, ratio1, p_value1 = linearRegression(df1, df2)
                    results2, ratio2, p_value2 = linearRegression(df2, df1)
                    if(symbol2=="SBIN" and symbol1=="AXISBANK"):
                        print(ratio1, ratio2)
                    if(ratio1<=ratio2):
                        if(p_value1<0.05):
                            pairs.append([industry1, symbol2, symbol1])

                    elif(ratio2<ratio1):
                        if(p_value2<0.05):
                            pairs.append([industry2, symbol1, symbol2])

                except Exception as e:
                    print(symbol1, symbol2, e)


    pairs = pd.DataFrame(pairs, columns=["Industry", "X", "Y"])
    pairs = pairs.drop_duplicates()

    if(export!=None):
        pairs.to_csv(f"{export}", index=False)

    return pairs

def get_std_err(df1, df2, time="Date"):

    # Extract the "Close" column
    df1 = df1["Close"]
    df2 = df2["Close"]

    df2_with_const = sm.add_constant(df2)

    # Ensure that both dataframes have the same index
    df1.index = df2_with_const.index

    # Fit the linear regression model
    model = sm.OLS(df1, df2_with_const).fit()

    beta_value = model.params[df2_with_const.columns[1]]

    residuals = model.resid

    # Standard deviation of residuals
    std_dev_residuals = residuals.std()

    std_err = residuals.iloc[-1]/std_dev_residuals
    
    return std_err

def get_LR_trades(pairs, start, end, lookback, datalocation, time="Date", export=None):

    temp = []
    for index, pair in pairs.iterrows():
        df1 = pd.read_csv(f"{datalocation}/{pair[1]}.csv")
        df2 = pd.read_csv(f"{datalocation}/{pair[2]}.csv")

        startIndex = df1.index[pd.to_datetime(df1[time]) == pd.to_datetime(start)].to_list()[0] - lookback
        endIndex = df1.index[pd.to_datetime(df1[time]) == pd.to_datetime(end)].to_list()[0] - lookback
        trades = []
        trade={}
        openTrade=False
        tradeType=""

        for i in range(startIndex, endIndex):
            new_df1 = df1[i:i+lookback]
            new_df2 = df2[i:i+lookback]
                
            std_err = get_std_err(new_df1, new_df2, time=time)

            if(std_err<=-2.5):
                entry = pd.to_datetime(new_df1[time].iloc[len(new_df1) - 1])
                formatted_entry = entry.strftime('%Y-%m-%d %H:%M:%S')

                trade["Stock X"] = pair[1]
                trade["Stock Y"] = pair[2]
                trade["type"]="long"
                trade["entry"]=formatted_entry
                openTrade=True
                tradeType="l"

            elif(std_err>=2.5):
                entry = pd.to_datetime(new_df1[time].iloc[len(new_df1) - 1])
                formatted_entry = entry.strftime('%Y-%m-%d %H:%M:%S')

                trade["Stock X"] = pair[1]
                trade["Stock Y"] = pair[2]
                trade["type"]="short"
                trade["entry"]=formatted_entry
                openTrade=True
                tradeType="s"

            elif(openTrade and tradeType=="s"):
                if(std_err>=3 or std_err<=1):
                    exit = pd.to_datetime(new_df1[time].iloc[len(new_df1) - 1])
                    formatted_exit = exit.strftime('%Y-%m-%d %H:%M:%S')
                    trade["exit"]=formatted_exit
                    openTrade=False
                    tradeType=""
                    trades.append(trade)
                    trade={}
                    
            elif(openTrade and tradeType=="l"):
                if(std_err<=-3 or std_err>=-1):
                    exit = pd.to_datetime(new_df1[time].iloc[len(new_df1) - 1])
                    formatted_exit = exit.strftime('%Y-%m-%d %H:%M:%S')
                    trade["exit"]=formatted_exit
                    openTrade=False
                    tradeType=""
                    trades.append(trade)
                    trade={}
        
        if(len(trades)!=0):
            temp.append(trades)

    if(export!=None):
        tradesExcel = []
        for pair in temp:
            for trade in pair:
                tradesExcel.append(trade)

        tradesExcel = pd.DataFrame(tradesExcel)
        tradesExcel.to_csv(f"{export}", index=False)

    return temp

def get_tracker(pairs, datalocation, start, end, time="Date", export=None):

    tracker = []
    for pair in pairs.iterrows():
        df1 = pd.read_csv(f"{datalocation}/{pair[1][1]}.csv")
        df2 = pd.read_csv(f"{datalocation}/{pair[1][2]}.csv")

        df1 = df1[pd.to_datetime(df1[time]).between(pd.to_datetime(start), pd.to_datetime(end))]
        df2 = df2[pd.to_datetime(df2[time]).between(pd.to_datetime(start), pd.to_datetime(end))]

        df1 = df1["Close"]
        df2 = df2["Close"]

        df1_with_const = sm.add_constant(df1)

        # Ensure that both dataframes have the same index
        df2.index = df1_with_const.index

        # Fit the linear regression model
        model = sm.OLS(df2, df1_with_const).fit()

        residuals = model.resid
        
        # Standard deviation of residuals
        std_dev_residuals = residuals.std()

        # Access Standard Error of Intercept
        standard_error_intercept = model.bse['const']

        errorRatio = standard_error_intercept/std_dev_residuals

        std_err = residuals.iloc[-1]/std_dev_residuals
        intercept = model.params['const']
        beta = model.params[df1_with_const.columns[1]]

        tracker.append([pair[1][1], pair[1][2], intercept, beta, std_dev_residuals, std_err])

    tracker = pd.DataFrame(tracker, columns=["X", "Y", "Intercept", "Beta", "std_dev_residuals", "std_err"])
    
    if(export!=None):
        tracker.to_excel(f"{export}", index=False)
    
    return tracker

def get_lot_size(lots, symbol):
  lot_size = lots.index[lots["Symbol"]==symbol]
  return lots.iloc[lot_size.to_list()[0]]["Jan-24"]

def get_txn_cost(buy, sell, quantity):
  buy_value = buy*quantity
  sell_value = sell*quantity
  turnover = buy_value+sell_value
  brokerage = buy_value*0.003 if buy_value*0.003 < 20 else 20
  brokerage+= sell_value*0.003 if sell_value*0.003 < 20 else 20
  stt = sell_value*(0.000125)
  txn_charges = (turnover)*0.000019
  sebi_charge = turnover*0.000001
  stamp_charges = buy_value*0.00002
  gst = (brokerage+sebi_charge+txn_charges)*1.18
  return brokerage+stt+txn_charges+sebi_charge+stamp_charges+gst

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