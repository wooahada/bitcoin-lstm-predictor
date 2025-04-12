# utils/realtime.py

import requests

def get_latest_price(symbol='BTCUSDT'):
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        params = {
            "category": "linear",  # 선물 기준
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if "result" in data and "list" in data["result"]:
            for item in data["result"]["list"]:
                if item["symbol"] == symbol:
                    return float(item["lastPrice"])
        return None
    except Exception as e:
        print("🔥 [ERROR] 실시간 가격 요청 실패:", str(e))
        return None
