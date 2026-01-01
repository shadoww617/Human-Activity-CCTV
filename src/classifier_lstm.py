import torch
import torch.nn as nn

class ActivityLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(99, 128, batch_first=True)
        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        out,_ = self.lstm(x)
        out = self.fc(out[:,-1])
        return out

labels = ["normal","fight","harassment","abnormal"]

model = ActivityLSTM()
model.load_state_dict(torch.load("models/lstm_activity.pth", map_location="cpu"))
model.eval()

def classify(sequence):
    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
    idx = pred.argmax().item()
    return labels[idx]
