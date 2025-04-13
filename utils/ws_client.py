
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
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # DEBUG에서 INFO로 변경하여 불필요한 로그 감소
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 웹소켓 디버그는 비활성화
websocket.enableTrace(False)

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

latest_ws_data = {
    "lastPrice": 0.0,
    "bid1Price": 0.0,
    "ask1Price": 0.0,
    "lastUpdateTime": 0,
    "lastMessageId": None  # 메시지 중복 체크용
}
_ws_instance = None
_ws_lock = threading.Lock()

def on_message(ws, message):
    global latest_ws_data
    try:
        data = json.loads(message)
        ticker_data = data.get("data", {})
        
        # 메시지 중복 체크
        message_id = f"{data.get('ts', 0)}_{data.get('cs', 0)}"
        with _ws_lock:
            if message_id == latest_ws_data["lastMessageId"]:
                return
            latest_ws_data["lastMessageId"] = message_id

        # timestamp 확인하여 이전 메시지 무시
        current_ts = data.get("ts", 0)
        with _ws_lock:
            if current_ts < latest_ws_data["lastUpdateTime"]:
                return

        # 가격 정보 추출 및 업데이트
        updates = {}
        if "lastPrice" in ticker_data:
            updates["lastPrice"] = float(ticker_data["lastPrice"])
        if "bid1Price" in ticker_data:
            updates["bid1Price"] = float(ticker_data["bid1Price"])
        if "ask1Price" in ticker_data:
            updates["ask1Price"] = float(ticker_data["ask1Price"])
        
        if updates:  # 업데이트할 가격 정보가 있는 경우에만
            with _ws_lock:
                latest_ws_data.update(updates)
                latest_ws_data["lastUpdateTime"] = current_ts
                logger.info(f"가격 업데이트: {dict((k, v) for k, v in latest_ws_data.items() if k != 'lastMessageId')}")

    except Exception as e:
        logger.error(f"메시지 파싱 오류: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket 오류 발생: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.warning(f"WebSocket 연결 종료됨: {close_status_code} - {close_msg}")
    if close_status_code != 1000:  # 정상 종료가 아닌 경우
        logger.info("5초 후 재연결 시도...")
        time.sleep(5)
        run_websocket_in_background()

def on_open(ws):
    logger.info("WebSocket 연결 성공")
    subscribe_msg = {
        "op": "subscribe",
        "args": ["tickers.BTCUSDT"]
    }
    ws.send(json.dumps(subscribe_msg))
    logger.info(f"구독 메시지 전송 완료")

def start_ws():
    global _ws_instance
    ws = websocket.WebSocketApp(
        "wss://stream.bybit.com/v5/public/linear",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    _ws_instance = ws
    ws.run_forever()

def run_websocket_in_background():
    thread = threading.Thread(target=start_ws)
    thread.daemon = True
    thread.start()
    return thread

def get_latest_ws_price():
    with _ws_lock:
        return latest_ws_data.get("lastPrice")

def get_latest_ws_prices():
    with _ws_lock:
        return dict(latest_ws_data)  # 복사본 반환하여 스레드 안전성 보장
