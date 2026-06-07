import os
import json
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset

if __name__ == '__main__':
    root = os.path.dirname(__file__)
    dataset_dir = os.path.abspath(os.path.join(root, '..', '..', 'data for model training', 'Plant_leave_diseases_dataset_without_augmentation'))
    model_dir = os.path.join(root, 'models')
    model_path = os.path.join(model_dir, 'disease_model.pth')
    class_names_path = os.path.join(model_dir, 'class_names.json')

    print('dataset_dir:', dataset_dir)
    print('model_path exists:', os.path.exists(model_path))
    print('class_names exists:', os.path.exists(class_names_path))

    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print('num_classes:', num_classes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(dataset_dir, transform=None)
    num_samples = len(full_dataset)
    print('num_samples:', num_samples)
    val_size = int(0.2 * num_samples)
    train_size = num_samples - val_size
    print('train_size, val_size:', train_size, val_size)

    splits = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_indices = splits[1].indices
    val_dataset = Subset(datasets.ImageFolder(dataset_dir, transform=transform), val_indices)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    print('val_correct:', correct)
    print('val_total:', total)
    print('val_accuracy:', correct / total * 100)
