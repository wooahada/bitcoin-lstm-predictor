import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_lstm_data(df, feature_col='close', sequence_length=60):
    """
    LSTM 학습을 위한 시계열 데이터 전처리.

    Parameters:
        df (DataFrame): Bybit에서 받은 원본 데이터프레임 (timestamp index 포함)
        feature_col (str): 사용할 컬럼명 (예: 'close')
        sequence_length (int): LSTM 시퀀스 길이 (default: 60)

    Returns:
        X_train, y_train, X_test, y_test, scaler
    """

    data = df[[feature_col]].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    # 80% train, 20% test 분리
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM input: (samples, timesteps, features)
    return X_train, y_train, X_test, y_test, scaler
