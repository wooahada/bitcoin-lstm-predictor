

import time
import pandas as pd
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API Key 및 API Secret 가져오기
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

# Pybit HTTP 객체 초기화
session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)

def get_bybit_historical_data(symbol='BTCUSDT', interval='60', limit=1000):
    try:
        # Bybit API v5 형식에 맞게 파라미터 설정
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        # Kline 데이터 요청
        logger.info(f"Bybit API 호출 시작: {params}")
        response = session.get_kline(**params)
        logger.info("API 응답 수신")
        
        # 응답 구조 확인 및 데이터 추출
        if not response or 'retCode' not in response:
            raise Exception(f"API 응답 형식 오류: {response}")
        
        if response['retCode'] != 0:
            raise Exception(f"API 오류 발생: {response['retMsg']}")
            
        result = response.get('result', {}).get('list', [])
        
        if not result:
            raise Exception("수신된 데이터가 없습니다")

        # DataFrame으로 변환 (V5 API 응답 구조에 맞게 수정)
        df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        
        # 시간 변환 및 정렬
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Seoul')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # 데이터 타입 변환
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        logger.info(f"데이터 처리 완료: {len(df)} 행")
        return df
    
    except Exception as e:
        logger.error(f"데이터 수집 실패: {str(e)}")
        raise

# 예시 사용법
df = get_bybit_historical_data(symbol='BTCUSDT', interval='60', limit=1000)

# 데이터 출력
print(df)
