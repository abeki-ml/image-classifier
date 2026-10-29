# Cats vs Dogs Image Classification Using CNN

## Overview
This project uses a Convolutional Neural Network (CNN) built with PyTorch to classify images into two categories: cats and dogs.

## Objective
To build and train a deep learning model that can identify whether an image contains a cat or a dog.

## Technologies Used
- Python
- PyTorch
- Torchvision
- Pillow (PIL)

## Dataset
The dataset contains cat and dog images organised into separate folders. Images are resized to 128 × 128 pixels and converted into tensors before being passed to the model.

The dataset is split into:
- 80% for training
- 20% for testing

## Model Architecture
The CNN consists of:
- Two convolutional layers
- ReLU activation functions
- Max pooling layers
- A fully connected hidden layer
- An output layer with two classes

## Results
- **Test accuracy:** 83.80%

The model was evaluated on a held-out test set. Its accuracy measures the percentage of test images classified correctly.

## Project Structure
```text
image-classifier/
├── data/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
├── models/
│   └── cats_dogs_cnn.pth
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py
```

### 3. Evaluate the model
```bash
python src/evaluate.py
```

### 4. Predict an image
```bash
python src/predict.py
```

## Future Improvements
- Improve classification accuracy.
- Handle corrupted images more reliably.
- Test the model on more diverse images.
- Explore data augmentation and transfer learning.

## Author
Fareedah Akindele
