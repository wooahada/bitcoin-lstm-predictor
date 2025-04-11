# test_bybit.py

import sys
import os

# ✅ 현재 파일 기준으로 루트 경로를 import 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.bybit_api import get_bybit_historical_data

# ✅ 테스트 실행
df = get_bybit_historical_data(interval='15', limit=1000)
print(df.head())
