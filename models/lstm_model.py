# models/lstm_model.py
import torch    
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
    
    #성능올리는 법

# ✅ 다중 피처용 LSTM 구조 변경 방법
# ✅ Dropout 추가해 정확도 향상하는 방법
# ✅ Bidirectional LSTM으로 변경하는 팁