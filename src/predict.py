
import torch
from torchvision import transforms
from PIL import Image
from model import CNN

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

model = CNN()
model.load_state_dict(
    torch.load("models/cats_dogs_cnn.pth", map_location="cpu")
)
model.eval()

image_path = "data/PetImages/Cat/1.jpg"

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image)
    predicted = output.argmax(dim=1).item()

classes = ["Cat", "Dog"]
print("Prediction:", classes[predicted])
