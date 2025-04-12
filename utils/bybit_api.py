import numpy as np
import pandas as pd
import requests
import time
import hashlib
import hmac

# Bybit API 호출을 위한 함수
def get_bybit_historical_data(symbol='BTCUSDT', interval='1d', limit=200, api_key=None, api_secret=None):
    url = 'https://api.bybit.com/v2/public/kline/list'
    
    # API 요청 파라미터 설정
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
        'api_key': api_key,
        'timestamp': str(int(time.time() * 1000))  # 타임스탬프 추가
    }

    # API 시크릿을 사용하여 서명 생성 (기본적으로 Bybit API는 서명 검증이 필요)
    if api_key and api_secret:
        params['sign'] = generate_signature(params, api_secret)

    # API 요청
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Bybit API 호출 실패: {response.status_code}")
    
    # JSON 응답에서 데이터 추출
    result = response.json().get('result', [])
    if not result:
        raise Exception("API로부터 데이터 없음")

    # 데이터프레임 생성
    df = pd.DataFrame(result, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    # UTC → KST 변환 및 정렬
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Seoul')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # 컬럼 데이터 타입 변환
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    return df

# 서명 생성 함수 (Bybit API에서 필요한 서명 생성)
def generate_signature(params, api_secret):
    sorted_params = sorted(params.items())
    query_string = '&'.join([f"{key}={value}" for key, value in sorted_params])
    payload = query_string + f"&api_secret={api_secret}"
    return hmac.new(payload.encode('utf-8'), api_secret.encode('utf-8'), hashlib.sha256).hexdigest()
