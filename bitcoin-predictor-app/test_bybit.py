# test_bybit.py

import sys
import os
import pandas as pd

# ✅ 현재 파일 기준으로 루트 경로를 import 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.bybit_api import get_bybit_historical_data

# ✅ 테스트 대상 인터벌
interval_map = {
    '5m': '5',
    '15m': '15',
    '1h': '60',
    '4h': '240'
}

# ✅ 각 인터벌별로 API 테스트
for label, interval in interval_map.items():
    print(f"\n⏱️ 테스트 중: {label} ({interval})")
    try:
        df = get_bybit_historical_data(interval=interval, limit=1000)
        print(f"✅ 데이터 수신 완료 ({label}): {df.shape}")
        print(df[['timestamp', 'close']].tail(3))
    except Exception as e:
        print(f"❌ 에러 발생 ({label}): {e}")
