# utils/ws_client.py

import json
import websocket
import threading
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib
import os
import sys
import time
import numpy as np  

# 폰트 설정
font_path = "assets/NotoSansCJKkr-Regular.otf"
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    matplotlib.rcParams['font.family'] = font_name
    matplotlib.rc('font', family=font_name)
    plt.rcParams['font.family'] = font_name  # 그래프 저장 시에도 적용
else:
    print("⚠️ [LOG] 폰트 파일이 없습니다.")

matplotlib.rcParams['axes.unicode_minus'] = False

latest_ws_data = {}

def on_message(ws, message):
    global latest_ws_data
    data = json.loads(message)
    try:
        ticker = data.get("data", {})
        latest_ws_data = {
            "lastPrice": float(ticker.get("lastPrice", 0)),
            "bid1Price": float(ticker.get("bid1Price", 0)),
            "ask1Price": float(ticker.get("ask1Price", 0))
        }
    except Exception as e:
        print("⚠️ WebSocket 메시지 파싱 오류:", e)

def on_error(ws, error):
    print("❌ WebSocket 오류:", error)

def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket 연결 종료")

def on_open(ws):
    print("✅ WebSocket 연결 성공")
    subscribe_msg = {
        "op": "subscribe",
        "args": ["tickers.BTCUSDT"]
    }
    ws.send(json.dumps(subscribe_msg))

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://stream.bybit.com/v5/public/linear",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

def run_websocket_in_background():
    thread = threading.Thread(target=start_ws)
    thread.daemon = True
    thread.start()

def get_latest_ws_price():  # ✅ 이전 버전 호환
    return latest_ws_data.get("lastPrice")

def get_latest_ws_prices():  # ✅ 새로운 다중 반환
    return latest_ws_data
