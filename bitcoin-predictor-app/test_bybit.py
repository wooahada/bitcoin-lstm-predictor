# test_bybit.py

from utils.bybit_api import get_bybit_historical_data

df = get_bybit_historical_data(symbol='BTCUSDT', interval='1d', limit=200)

print(df.head())
print(df.tail())