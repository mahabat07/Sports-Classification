import json
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from PIL import Image
from io import BytesIO

app = FastAPI(title="Sports Classification API")

device = "cuda" if torch.cuda.is_available() else "cpu"


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 100)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


with open("classes.json", "r") as f:
    classes = json.load(f)


model = MyModel().to(device)
model.load_state_dict(torch.load("sports_model.pth",map_location=device,weights_only=True))
model.eval()
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        image = transform(image)
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)

            confidence, predicted = torch.max(
                probabilities, dim=1
            )

        class_name = classes[predicted.item()]

        return {
            "class_name": class_name,
            "confidence": round(confidence.item() * 100, 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки изображения: {str(e)}"
        )

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)