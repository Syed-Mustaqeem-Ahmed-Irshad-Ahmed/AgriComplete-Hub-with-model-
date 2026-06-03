import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Configurations
DATASET_DIR = r"C:\Users\musta\Downloads\Main\agricomplete-hub-main\agricomplete-hub-main\data for model training\Plant_leave_diseases_dataset_without_augmentation"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 3
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading dataset...")
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
class_names = full_dataset.classes
print(f"Found {len(class_names)} classes.")

# Save class names for inference
with open(os.path.join(MODEL_DIR, 'class_names.json'), 'w') as f:
    json.dump(class_names, f)

# For quick demonstration, let's use a very small subset of the dataset if it's too large.
# E.g., we'll just train on a 10% subset to save time if using CPU.
subset_size = int(0.1 * len(full_dataset))
remaining_size = len(full_dataset) - subset_size
subset_dataset, _ = random_split(full_dataset, [subset_size, remaining_size])

train_size = int(0.8 * len(subset_dataset))
val_size = len(subset_dataset) - train_size
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Building model...")
# Base model: MobileNetV2
model = models.mobilenet_v2(pretrained=True)
# Freeze base model
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Step {i+1}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
            
    print(f"Epoch {epoch+1} completed. Loss: {running_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}%")

# Save model weights
model_path = os.path.join(MODEL_DIR, 'disease_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model weights saved to {model_path}")
