

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import torch  
import torch.nn as nn
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import argparse
import random
import os
import numpy as np


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 

seed_everything()


parser = argparse.ArgumentParser(description='Finetune EfficientNet')

parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transforms for the train and validation sets
transform_train = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load the train and validation sets
train_dataset = datasets.ImageFolder('<path to train data>', transform=transform_train)

val_dataset = datasets.ImageFolder('<path to val data>', transform=transform_val) 


# Create the train and validation loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)


for param in model.parameters():
    param.requires_grad = True

for param in model._fc.parameters():
    param.requires_grad = True

model._fc = nn.Linear(model._fc.in_features, 2)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


best_acc = 0.0
best_loss = float('inf')

num_epochs = args.epochs

# Train the model for a specified number of epochs
for epoch in tqdm(range(num_epochs)): 
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

    # Validation loop
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    val_acc = 0
    val_loss = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data) 
            val_loss += loss.item() * inputs.size(0)
    epoch_acc = val_acc.double() / (len(val_loader.dataset))
    epoch_loss = val_loss / len(val_loader.dataset)

    # Print the epoch loss and accuracy
    print('Epoch [{}/{}], Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
        epoch+1, num_epochs, epoch_loss, epoch_acc))

    
    if epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss < best_loss):
        best_acc = epoch_acc
        best_loss = epoch_loss
        torch.save(model.state_dict(), f'<path to save output model>.pth') 





