import torch
import torch.nn as nn
from fastapi import FastAPI

app = FastAPI()

# Model class (same as before)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = Model()
model.load_state_dict(torch.load("cancer_model.pth"))
model.eval()

# Root
@app.get("/")
def home():
    return {"message": "AI Model API Running"}

# Prediction endpoint
from pydantic import BaseModel

class InputData(BaseModel):
    data: list

@app.post("/predict")
def predict(input: InputData):
    
    x = torch.tensor([input.data], dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(x).item()
    
    return {
        "prediction": prediction,
        "class": int(prediction > 0.5)
    }