import utility_functions as uf
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

symbols = pd.read_csv("Symbols50.csv")
uf.fetch_data("Historical Data", start="2022-01-01", end="2024-02-25", interval="1d", symbols=symbols["Symbol"])
pairs = uf.get_lr_pairs(symbols, "Historical Data", "23-02-2024", 200)
print(uf.get_tracker(pairs, "Historical Data", "2024-02-23", 200))
# print(uf.temp_funct(symbols, "Historical Data", 200, start="2024-02-15", end="2024-02-22"))

# uf.temp_lr_pairs(symbols, "Historical Data", "2024-01-01", 200)