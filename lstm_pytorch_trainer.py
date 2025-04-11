# train_lstm_multi.py

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from utils.bybit_api import get_bybit_historical_data
from utils.preprocess import preprocess_lstm_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.lstm_model import LSTMRegressor  # ✅ 모델 클래스만 import

# 하이퍼파라미터
SEQ_LEN = 60
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
INTERVALS = {
    '5m': '5',
    '15m': '15',
    '1h': '60',
    '4h': '240'
}

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_and_save_model(interval_name, bybit_interval):
        print(f"\n🚀 [{interval_name}] 학습 시작 - API interval: {bybit_interval}")

        try:
            df = get_bybit_historical_data(interval=bybit_interval, limit=1000)
            print(f"✅ [{interval_name}] 데이터 수집 성공 - shape: {df.shape}")
        except Exception as e:
            print(f"❌ [{interval_name}] 데이터 수집 실패: {str(e)}")
            return

        print(f"\n📈 [{interval_name}] 데이터로 학습 시작")

        # 1. 데이터 수집
        df = get_bybit_historical_data(interval=bybit_interval, limit=1000)

        # 2. 전처리
        X_train, y_train, X_test, y_test, scaler = preprocess_lstm_data(df, sequence_length=SEQ_LEN)

        # 3. Tensor 변환
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

        # 4. 모델 정의
        model = LSTMRegressor()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # 5. 학습
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[{interval_name}] Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

        # 6. 모델 저장
        os.makedirs("models", exist_ok=True)
        model_path = f"models/lstm_model_{interval_name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"✅ [{interval_name}] 모델 저장 완료 → {model_path}")

        # ✅ 7. 정확도 평가 (함수 내부!)
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

        y_pred_real = scaler.inverse_transform(y_pred)
        y_test_real = scaler.inverse_transform(y_test)

        mae = mean_absolute_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))  # 🔧 수정된 줄
        r2 = r2_score(y_test_real, y_pred_real)


        print(f"\n📊 [{interval_name}] 정확도 평가 결과")
        print(f"📉 MAE (절대 오차 평균): {mae:.2f}")
        print(f"📉 RMSE (제곱 평균 오차): {rmse:.2f}")
        print(f"📈 R² Score (결정계수): {r2:.4f}")

# 🔥 메인 실행부 추가 (반드시 파일 끝에!)
if __name__ == "__main__":
    print("🔥 FILE EXECUTED\n")

    INTERVALS = {
        '5m': '5',
        '15m': '15',
        '1h': '60',
        '4h': '240'
    }

    for name, bybit_code in INTERVALS.items():
        train_and_save_model(name, bybit_code)
