import utility_functions as uf
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# symbols = pd.read_csv("Symbols50.csv")
# symbols = symbols["Symbol"]
# uf.fetch_data("Historical Data", start="2023-12-01", end="2024-01-04", interval="15m", symbols=symbols)
# symbols = pd.read_csv("symbols50.csv")
# pairs = uf.get_LR_pairs(symbols, "Historical Data", start="01-12-2023", end="30-12-2023", time="Datetime", export="Pair Trading/Intraday Pairs.csv")
pairs = pd.read_csv("Pair Trading/Intraday Pairs.csv")
tracker = uf.get_tracker(pairs,"Historical Data",start="27-12-2023", end="04-01-2024", time="Datetime", export="Temp Tracker.xlsx")
# uf.get_LR_trades(pairs, start="2024-01-03 09:15:00", end="2024-01-03 15:15:00", lookback=125, datalocation="Historical Data", time="Datetime", export="Temp.csv")

