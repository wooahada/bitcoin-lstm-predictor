# test_bybit.py
from utils.bybit_api import get_bybit_historical_data

df = get_bybit_historical_data(interval='15', limit=1000)
print(df.head())