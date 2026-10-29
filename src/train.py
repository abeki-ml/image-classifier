import torch.nn as nn
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch

import torch.optim as optim
from PIL import Image
from model import CNN

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

print("Total images:", len(train_data))
print("Class names:", full_data.classes)


images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

model = CNN()
output = model(images)

print(output.shape)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    average_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/5], Loss: {average_loss:.4f}")


    
torch.save(model.state_dict(), "models/cats_dogs_cnn.pth")
print("Model saved successfully!")













