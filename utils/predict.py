# utils/predict.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib import font_manager

from utils.bybit_api import get_bybit_historical_data
from utils.preprocess import preprocess_lstm_data
from models.lstm_model import LSTMRegressor

# ✅ 한글 폰트 설정 (배포 환경 대응)
try:
    font_path = "assets/micross.ttf"
    if os.path.exists(font_path):
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        matplotlib.rcParams['font.family'] = font_name
        print(f"✅ [LOG] predict.py에서 폰트 적용 완료: {font_name}")
    else:
        print("⚠️ [LOG] predict.py에서 폰트 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"⚠️ [LOG] predict.py 폰트 설정 실패: {e}")

matplotlib.rcParams['axes.unicode_minus'] = False

def predict_lstm_price(interval='5m', model_dir='models', steps=1):
    try:
        print(f"✅ [LOG] 예측 함수 진입: interval={interval}, steps={steps}")

        interval_map = {
            '5m': '5', '15m': '15', '1h': '60', '4h': '240'
        }

        if interval not in interval_map:
            print("❌ [LOG] 지원되지 않는 시간 단위")
            return None, None, None

        df = get_bybit_historical_data(interval=interval_map[interval], limit=1000)
        if df.empty:
            print("❌ [LOG] 수신된 데이터 없음")
            return None, None, None

        print("✅ [LOG] 데이터 수신 완료:", df.shape)

        X_train, y_train, X_test, y_test, scaler = preprocess_lstm_data(df, sequence_length=60)
        input_seq = X_test[-1:]
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)

        model_path = os.path.join(model_dir, f"lstm_model_{interval}.pt")
        if not os.path.exists(model_path):
            print("❌ [LOG] 모델 파일 없음")
            return None, None, None

        model = LSTMRegressor()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        print("🧠 [LOG] 모델 로딩 성공")

        with torch.no_grad():
            for _ in range(steps):
                prediction = model(input_tensor)
                input_tensor = torch.cat((input_tensor[:, 1:, :], prediction.unsqueeze(1)), dim=1)

        predicted_price = scaler.inverse_transform(prediction.numpy())[0][0]
        last_close = df['close'].iloc[-1]

        print("🎯 [LOG] 예측 완료")
        return predicted_price, last_close, df

    except Exception as e:
        import traceback
        print("🔥 [ERROR] 예외 발생:", str(e))
        traceback.print_exc()  # 👉 터미널에 전체 트레이스 출력
        return None, None, None
