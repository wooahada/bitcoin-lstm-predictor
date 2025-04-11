# utils/bybit_api.py

import numpy as np
import pandas as pd
import requests

def get_bybit_historical_data(symbol='BTCUSDT', interval='1d', limit=1000):
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Bybit API 호출 실패: {response.status_code}")

    result = response.json().get('result', {}).get('list', [])
    if not result:
        raise Exception("API로부터 데이터 없음")

    df = pd.DataFrame(result, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    # ✅ UTC → KST 변환 및 정렬
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Seoul')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    return df
