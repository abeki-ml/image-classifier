import torch
from torch.utils.data import random_split, DataLoader
from PIL import Image
from torchvision import datasets, transforms

from model import CNN


model = CNN()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False



full_data = datasets.ImageFolder(
    root = "data/PetImages",
    transform = transform,
    is_valid_file = is_valid_image
)

train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size

train_data, test_data = random_split(
    full_data,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)    

)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)


model.load_state_dict(
    torch.load("models/cats_dogs_cnn.pth", map_location="cpu")
)
model.eval()


correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total  * 100
print(f"Test Accuracy: {accuracy:.2f}%")

