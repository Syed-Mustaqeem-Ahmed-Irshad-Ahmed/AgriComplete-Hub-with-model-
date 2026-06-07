import argparse
import json
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


def get_default_dataset_dir():
    root = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(
        root,
        '..',
        '..',
        'data for model training',
        'Plant_leave_diseases_dataset_without_augmentation'
    ))


def get_augmented_dataset_dir():
    root = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(
        root,
        '..',
        '..',
        'data for model training',
        'Plant_leave_diseases_dataset_with_augmentation'
    ))


def parse_args():
    parser = argparse.ArgumentParser(description='Train the plant disease model.')
    parser.add_argument('--dataset-dir', default=get_default_dataset_dir(), help='Path to the ImageFolder dataset.')
    parser.add_argument('--use-augmented-dir', action='store_true', help='Use the augmented dataset directory if available.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation ratio.')
    parser.add_argument('--model-dir', default=os.path.join(os.path.dirname(__file__), 'models'), help='Directory to save the trained model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    return parser.parse_args()


def make_transforms(img_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def build_model(num_classes, device):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)


def compute_class_weights(targets, num_classes):
    counts = Counter(targets)
    class_counts = [counts[i] for i in range(num_classes)]
    total = float(sum(class_counts))
    weights = [total / (c if c > 0 else 1) for c in class_counts]
    return torch.tensor(weights, dtype=torch.float32), [weights[t] for t in targets]


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (inputs, labels) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if step % 20 == 0:
            print(f'Epoch {epoch} Step {step}/{len(loader)} Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%')

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_dir = args.dataset_dir
    if args.use_augmented_dir:
        augmented = get_augmented_dataset_dir()
        if os.path.isdir(augmented):
            dataset_dir = augmented
            print(f'Using augmented dataset directory: {dataset_dir}')
        else:
            print(f'Augmented dataset directory not found, using: {dataset_dir}')

    print(f'Using dataset directory: {dataset_dir}')
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f'Dataset directory not found: {dataset_dir}')

    os.makedirs(args.model_dir, exist_ok=True)

    train_transform, val_transform = make_transforms(args.img_size)
    full_dataset = datasets.ImageFolder(dataset_dir, transform=None)
    num_classes = len(full_dataset.classes)
    print(f'Found {num_classes} classes.')

    class_names = full_dataset.classes
    class_names_path = os.path.join(args.model_dir, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f'Saved class names to {class_names_path}')

    num_samples = len(full_dataset)
    val_size = int(args.val_split * num_samples)
    train_size = num_samples - val_size

    indices = list(range(num_samples))
    torch.manual_seed(args.seed)
    random_split = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_indices = random_split[0].indices
    val_indices = random_split[1].indices

    train_dataset = Subset(datasets.ImageFolder(dataset_dir, transform=train_transform), train_indices)
    val_dataset = Subset(datasets.ImageFolder(dataset_dir, transform=val_transform), val_indices)

    class_weights_tensor, sample_weights = compute_class_weights([full_dataset.targets[i] for i in train_indices], num_classes)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    model = build_model(num_classes, device)
    class_weights_tensor = class_weights_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0
    best_model_path = os.path.join(args.model_dir, 'disease_model.pth')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f'Epoch {epoch}/{args.epochs} finished:')
        print(f'  Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%')
        print(f'  Val   loss: {val_loss:.4f}, Val   acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'  New best model saved to {best_model_path} (val_acc={best_acc:.2f}%)')

    print(f'Training complete. Best validation accuracy: {best_acc:.2f}%')
    print(f'Model and class names saved to: {args.model_dir}')


if __name__ == '__main__':
    main()
