# # models/lstm_model.py
# import torch    
# import torch.nn as nn

# class LSTMRegressor(nn.Module):
#     def __init__(self, input_size=1, hidden_size=50, num_layers=2):
#         super(LSTMRegressor, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])
    
#     #성능올리는 법

# # ✅ 다중 피처용 LSTM 구조 변경 방법
# # ✅ Dropout 추가해 정확도 향상하는 방법
# # ✅ Bidirectional LSTM으로 변경하는 팁

# models/lstm_model.py
import torch    
import torch.nn as nn
import os

class LSTMRegressor(nn.Module):
    """Bitcoin price prediction using LSTM"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Ensure input tensor is float32
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype != torch.float32:
            x = x.float()
            
        # Run LSTM
        out, _ = self.lstm(x)
        # Use only the last output for prediction
        return self.fc(out[:, -1, :])
    
    #성능올리는 법

# ✅ 다중 피처용 LSTM 구조 변경 방법
# ✅ Dropout 추가해 정확도 향상하는 방법
# ✅ Bidirectional LSTM으로 변경하는 팁    @staticmethod
    def load_model(model_path):
        """Load model from checkpoint file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMRegressor()
        
        try:
            # Load the state dict
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")