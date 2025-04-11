# utils/bybit_api.py

import requests
import pandas as pd

def get_bybit_historical_data(symbol='BTCUSDT', interval='1d', limit=200):
    """
    Bybit API를 사용해 히스토리컬 가격 데이터를 불러옴.

    Parameters:
        symbol (str): 코인 심볼 (기본: BTCUSDT)
        interval (str): 데이터 간격 (1d, 1h, 15, etc.)
        limit (int): 가져올 데이터 수 (최대 1000)

    Returns:
        DataFrame: 시가, 고가, 저가, 종가, 거래량 등의 데이터를 포함한 데이터프레임
    """
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

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    return df
