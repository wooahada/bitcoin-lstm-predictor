# utils/predict.py (개선된 버전)

import os
import torch
import numpy as np
from utils.bybit_api import get_bybit_historical_data
from utils.preprocess import preprocess_lstm_data
from models.lstm_model import LSTMRegressor

import matplotlib.pyplot as plt


def plot_prediction(df, predicted_price, interval='15m'):
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(df['close'][-100:], label='실제 가격', linewidth=2)
        plt.axhline(y=predicted_price, color='red', linestyle='--', label='예측 가격')
        plt.title(f'BTCUSDT {interval} 예측 결과', fontsize=14)
        plt.xlabel('시간')
        plt.ylabel('가격 (USDT)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        input("📊 그래프 창을 닫으면 계속 진행됩니다. 엔터를 눌러주세요.")
    except Exception as e:
        print("🔥 [ERROR] 그래프 시각화 실패:", str(e))


def predict_lstm_price(interval='5m', model_dir='models', steps=1):
    try:
        print(f"✅ [LOG] 예측 함수 진입: interval={interval}, steps={steps}")

        interval_map = {
            '5m': '5',
            '15m': '15',
            '1h': '60',
            '4h': '240'
        }

        if interval not in interval_map:
            print("❌ [LOG] 지원되지 않는 시간 단위")
            return None, None, None

        print("📡 [LOG] Bybit 데이터 요청 중...")
        df = get_bybit_historical_data(interval=interval_map[interval], limit=1000)

        if df is None:
            print("❌ [LOG] df is None")
            return None, None, None
        if df.empty:
            print("❌ [LOG] df is empty")
            return None, None, None

        print("✅ [LOG] 데이터 수신 완료:", df.shape)

        X_train, y_train, X_test, y_test, scaler = preprocess_lstm_data(df, sequence_length=60)
        input_seq = X_test[-1:]
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)

        model = LSTMRegressor()
        model_path = os.path.join(model_dir, f"lstm_model_{interval}.pt")

        print("📦 [LOG] 모델 경로 확인:", model_path)

        if not os.path.exists(model_path):
            print("❌ [LOG] 모델 파일 없음")
            return None, None, None

        model.load_state_dict(torch.load(model_path))
        model.eval()

        print("🧠 [LOG] 모델 로딩 성공")

        with torch.no_grad():
            for _ in range(steps):
                prediction = model(input_tensor)
                input_tensor = torch.cat((input_tensor[:, 1:, :], prediction.unsqueeze(1)), dim=1)

        predicted_scaled = prediction.numpy()
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
        last_close = df['close'].iloc[-1]

        print("🎯 [LOG] 예측 완료")
        return predicted_price, last_close, df

    except Exception as e:
        import traceback
        print("🔥 [ERROR] 예외 발생:", str(e))
        traceback.print_exc()
        return None, None, None