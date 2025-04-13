# # lstm_pytorch_trainer.py

# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import matplotlib.dates as mdates
# from matplotlib import font_manager
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset

# from utils.bybit_api import get_bybit_historical_data
# from utils.preprocess import preprocess_lstm_data
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.metrics import mean_squared_error

# from models.lstm_model import LSTMRegressor  # ✅ 외부 모델 파일에서 import

# # ✅ 한글 폰트 설정 (배포 환경에서도 안전하게)
# try:
#     font_path = "assets/micross.ttf"
#     if os.path.exists(font_path):
#         font_prop = font_manager.FontProperties(fname=font_path)
#         font_name = font_prop.get_name()
#         matplotlib.rcParams['font.family'] = font_name
#     else:
#         font_prop = None
# except Exception as e:
#     print(f"⚠️ [LOG] 폰트 설정 실패: {e}")
#     font_prop = None

# matplotlib.rcParams['axes.unicode_minus'] = False

# # 하이퍼파라미터
# SEQ_LEN = 60
# BATCH_SIZE = 32
# EPOCHS = 30
# LR = 0.001

# def train_and_save_model(interval_name, bybit_interval):
#     print(f"\n📈 [{interval_name}] 데이터로 학습 시작")

#     # 1. 데이터 수집
#     df = get_bybit_historical_data(interval=bybit_interval, limit=1000)
#     if df is None or df.empty:
#         print(f"❌ [{interval_name}] 데이터 수집 실패")
#         return

#     # 2. 전처리
#     X_train, y_train, X_test, y_test, scaler = preprocess_lstm_data(df, sequence_length=SEQ_LEN)

#     # y_test shape 보정 (scaler 오류 방지)
#     if y_test.ndim == 1:
#         y_test = y_test.reshape(-1, 1)

#     # 3. Tensor 변환
#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     y_train = torch.tensor(y_train, dtype=torch.float32)
#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

#     # 4. 모델 정의
#     model = LSTMRegressor()
#     print(model)  # ⭐⭐⭐⭐⭐ 모델 구조 출력
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     # 5. 학습
#     model.train()
#     for epoch in range(EPOCHS):
#         total_loss = 0
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             output = model(batch_X)
#             loss = criterion(output, batch_y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"[{interval_name}] Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

#     # 6. 모델 저장
#     os.makedirs("models", exist_ok=True)
#     model_path = f"models/lstm_model_{interval_name}.pt"
#     torch.save(model.state_dict(), model_path)
#     print(f"✅ [{interval_name}] 모델 저장 완료 → {model_path}")

#     # 7. 정확도 평가
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

#     y_pred_real = scaler.inverse_transform(y_pred)
#     y_test_real = scaler.inverse_transform(y_test)

#     mae = mean_absolute_error(y_test_real, y_pred_real)
#     rmse = mean_squared_error(y_test_real, y_pred_real) ** 0.5  # 수동으로 루트 계산
#     r2 = r2_score(y_test_real, y_pred_real)

#     print(f"\n📊 [{interval_name}] 정확도 평가 결과")
#     print(f"📉 MAE: {mae:.2f}")
#     print(f"📉 RMSE: {rmse:.2f}")
#     print(f"📈 R² Score: {r2:.4f}")

#     # 8. 시각화 (선택적으로)
#     try:
#         fig, ax = plt.subplots(figsize=(12, 5))
#         ax.plot(df.index[-100:], df['close'][-100:], label='실제 가격', linewidth=2)
#         ax.plot(df.index[-1], y_pred_real[-1], 'ro', label='예측 가격')
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
#         fig.autofmt_xdate()
#         ax.set_title(f'{interval_name} 예측 결과', fontsize=14, fontproperties=font_prop)
#         ax.set_xlabel('시간', fontproperties=font_prop)
#         ax.set_ylabel('가격 (USDT)', fontproperties=font_prop)
#         ax.legend(prop=font_prop)
#         ax.grid(True)
#         plt.tight_layout()
#         plt.show()
#     except Exception as vis_err:
#         print(f"⚠️ [LOG] 시각화 실패: {vis_err}")

# # 🔥 메인 실행부
# if __name__ == "__main__":
#     print("🔥 FILE EXECUTED\n")

#     INTERVALS = {
#         '5m': '5',
#         '15m': '15',
#         '1h': '60',
#         '4h': '240'
#     }

#     for name, bybit_code in INTERVALS.items():
#         train_and_save_model(name, bybit_code)

# lstm_pytorch_trainer.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib import font_manager
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.bybit_api import get_bybit_historical_data
from utils.preprocess import preprocess_lstm_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error

from models.lstm_model import LSTMRegressor  # ✅ 외부 모델 파일에서 import

# ✅ 한글 폰트 설정 (배포 환경에서도 안전하게)
try:
    font_path = "assets/micross.ttf"
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        matplotlib.rcParams['font.family'] = font_name
    else:
        font_prop = None
except Exception as e:
    print(f"⚠️ [LOG] 폰트 설정 실패: {e}")
    font_prop = None

matplotlib.rcParams['axes.unicode_minus'] = False

# 하이퍼파라미터
SEQ_LEN = 60
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

def train_and_save_model(interval_name, bybit_interval):
    print(f"\n📈 [{interval_name}] 데이터로 학습 시작")

    # 1. 데이터 수집 - Bybit API v5 형식에 맞게 interval 수정
    INTERVALS = {
        '5m': '5',
        '15m': '15',
        '1h': '60',
        '4h': '240'
    }
    
    df = get_bybit_historical_data(interval=INTERVALS[interval_name], limit=1000)
    if df is None or df.empty:
        print(f"❌ [{interval_name}] 데이터 수집 실패")
        return

    # 2. 전처리
    X_train, y_train, X_test, y_test, scaler = preprocess_lstm_data(df, sequence_length=SEQ_LEN)

    # y_test shape 보정 (scaler 오류 방지)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # 3. Tensor 변환
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    # 4. 모델 정의
    model = LSTMRegressor()
    print(model)  # ⭐⭐⭐⭐⭐ 모델 구조 출력
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

    # 7. 정확도 평가
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    y_pred_real = scaler.inverse_transform(y_pred)
    y_test_real = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = mean_squared_error(y_test_real, y_pred_real) ** 0.5  # 수동으로 루트 계산
    r2 = r2_score(y_test_real, y_pred_real)

    print(f"\n📊 [{interval_name}] 정확도 평가 결과")
    print(f"📉 MAE: {mae:.2f}")
    print(f"📉 RMSE: {rmse:.2f}")
    print(f"📈 R² Score: {r2:.4f}")

    # 8. 시각화 (선택적으로)
    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index[-100:], df['close'][-100:], label='실제 가격', linewidth=2)
        ax.plot(df.index[-1], y_pred_real[-1], 'ro', label='예측 가격')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()
        ax.set_title(f'{interval_name} 예측 결과', fontsize=14, fontproperties=font_prop)
        ax.set_xlabel('시간', fontproperties=font_prop)
        ax.set_ylabel('가격 (USDT)', fontproperties=font_prop)
        ax.legend(prop=font_prop)
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as vis_err:
        print(f"⚠️ [LOG] 시각화 실패: {vis_err}")

# 🔥 메인 실행부
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
