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
# from tvDatafeed import TvDatafeed, Interval

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

def linearRegression(y, x):

    # Add a constant term to the independent variable
    x_with_const = sm.add_constant(x)

    # Ensure that both dataframes have the same index
    y.index = x_with_const.index

    # Fit the linear regression model
    model = sm.OLS(y, x_with_const).fit()

    residuals = model.resid

    # Standard deviation of residuals
    std_dev_residuals = residuals.std()

    # Access Standard Error of Intercept
    standard_error_intercept = model.bse['const']

    errorRatio = standard_error_intercept/std_dev_residuals

    std_err = residuals.iloc[-1]/std_dev_residuals

    result = adfuller(residuals)

    # Extracting test statistics and p-value
    adf_statistic = result[0]
    p_value = result[1]

    # return model.summary(), errorRatio, p_value, std_err
    return errorRatio, p_value

def get_tracker(pairs, datalocation, start, lookback, time="Date", export=None):

    tracker = []
    for pair in pairs.iterrows():
        x = pd.read_csv(f"{datalocation}/{pair[1][1]}.csv")
        y = pd.read_csv(f"{datalocation}/{pair[1][2]}.csv")

        startIndex = x.index[pd.to_datetime(x[time]) == pd.to_datetime(start)].to_list()[0] - lookback
        endIndex = x.index[pd.to_datetime(x[time]) == pd.to_datetime(start)].to_list()[0]

        x = x[startIndex:endIndex+1]
        y = y[startIndex:endIndex+1]

        x = x["Close"]
        y = y["Close"]

        x_with_const = sm.add_constant(x)

        # Ensure that both dataframes have the same index
        y.index = x_with_const.index

        # Fit the linear regression model
        model = sm.OLS(y, x_with_const).fit()

        residuals = model.resid
        
        # Standard deviation of residuals
        std_dev_residuals = residuals.std()

        # Access Standard Error of Intercept
        standard_error_intercept = model.bse['const']

        errorRatio = standard_error_intercept/std_dev_residuals

        std_err = residuals.iloc[-1]/std_dev_residuals
        intercept = model.params['const']
        beta = model.params[x_with_const.columns[1]]

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

def get_lr_entries(symbols, dataLocation, lookback, start, end):

    startIndex = pd.to_datetime(start)
    endIndex = pd.to_datetime(end)
    currentDate=startIndex
    trades = []

    while(currentDate >= startIndex and currentDate <= endIndex):
        pairs = get_lr_pairs(symbols, dataLocation, str(currentDate), lookback)

        tracker = get_tracker(pairs, dataLocation, currentDate, lookback)

        temp = get_lr_trades(tracker, currentDate)

        for trade in temp:
            trades.append(trade)

        currentDate = currentDate + timedelta(days=1)

    return pd.DataFrame(trades, columns=["entry", "Stock X", "Stock Y", "type", "std_err"])

def get_lr_pairs(symbols, dataLocation, startDate, lookback, time="Date"):

    pairs = []
    for row1 in symbols.iterrows():
        for row2 in symbols.iterrows():
            industry1 = row1[1][1]
            industry2 = row2[1][1]
            symbol1 = row1[1][2]
            symbol2 = row2[1][2]

            if(industry1 == industry2 and symbol1 != symbol2 and [symbol2, symbol1] not in pairs):
                
                try:
                    xDf = pd.read_csv(f"{dataLocation}/{symbol1}.csv")
                    yDf = pd.read_csv(f"{dataLocation}/{symbol2}.csv")
                    
                    startIndex = xDf.index[pd.to_datetime(xDf[time]) == pd.to_datetime(startDate)].to_list()[0] - lookback
                    endIndex = xDf.index[pd.to_datetime(xDf[time]) == pd.to_datetime(startDate)].to_list()[0]

                    x = xDf[startIndex:endIndex+1]
                    y = yDf[startIndex:endIndex+1]

                    # Extract the "Close" column
                    x = x["Close"]
                    y = y["Close"]

                    ratio1, p_value1 = linearRegression(y, x)
                    ratio2, p_value2 = linearRegression(x, y)

                    if(ratio1<=ratio2):
                        if(p_value1<0.05):
                            pairs.append([industry1, symbol1, symbol2])

                    elif(ratio2<ratio1):
                        if(p_value2<0.05):
                            pairs.append([industry2, symbol2, symbol1])

                except Exception as e:
                    if(e!=IndexError):
                        print(f"{symbol1} - {symbol2} : {e}")
    
    pairs = pd.DataFrame(pairs, columns=["Industry", "X", "Y"])
    pairs = pairs.drop_duplicates()

    return pairs

def get_lr_trades(tracker, date):

    trades = []
    for index, row in tracker.iterrows():
        trade={}
        if(tracker.iloc[index]["std_err"]<-2.5):
            trade["entry"]=date
            trade["Stock X"] = tracker.iloc[index]["X"]
            trade["Stock Y"] = tracker.iloc[index]["Y"]
            trade["type"]="long"
            trade["std_err"]=tracker.iloc[index]["std_err"]
            trades.append(trade)

        if(tracker.iloc[index]["std_err"]>2.5):
            trade["entry"]=date
            trade["Stock X"] = tracker.iloc[index]["X"]
            trade["Stock Y"] = tracker.iloc[index]["Y"]
            trade["type"]="short"
            trade["std_err"]=tracker.iloc[index]["std_err"]
            trades.append(trade)

    return trades
