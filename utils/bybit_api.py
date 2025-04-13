import time
import pandas as pd
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API Key 및 API Secret 가져오기
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

# Pybit HTTP 객체 초기화
session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)

# Bybit의 역사적인 데이터 가져오는 함수
def get_bybit_historical_data(symbol='BTCUSDT', interval='1h', limit=200):
    try:
        # Kline 데이터 요청
        response = session.get_kline(symbol=symbol, interval=interval, limit=limit)
        
        # 응답 상태 코드 확인
        if not response or 'result' not in response:
            raise Exception(f"API 호출 실패: {response}")
        
        # 데이터 추출
        result = response['result']
        
        if not result:
            print("데이터 없음: 확인이 필요합니다.")
            raise Exception("API로부터 데이터 없음")

        # DataFrame으로 변환
        df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        
        # UTC → KST 변환 및 정렬
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Seoul')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # 컬럼 값 숫자 타입으로 변환
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        return df
    
    except Exception as e:
        print(f"예외 발생: {e}")
        raise Exception(f"예외 발생: {e}")


# 예시 사용법
df = get_bybit_historical_data(symbol='BTCUSDT', interval='1h', limit=200)

# 데이터 출력
print(df)
