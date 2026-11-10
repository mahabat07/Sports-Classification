import json
import torch
import torch.nn as nn
import streamlit as st
from torchvision import transforms
from PIL import Image


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
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


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MyModel().to(device)

    model.load_state_dict(
        torch.load(
            "sports_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()
    return model, device


model, device = load_model()
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


st.title("Sports Classification")
st.write("Загрузите изображение, чтобы определить вид спорта.")

uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Загруженное изображение")

    if st.button("Определить спорт"):

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)

            probabilities = torch.softmax(output, dim=1)

            confidence, predicted = torch.max(
                probabilities, dim=1
            )

        class_name = classes[predicted.item()]

        st.success(f"Вид спорта: {class_name}")

        st.write(
            f"Уверенность модели: {confidence.item() * 100:.2f}%"
        )