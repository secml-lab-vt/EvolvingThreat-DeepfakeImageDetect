
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
    
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--input_path', type=str, default=None)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(root=args.input_path, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(test_dataset.classes)
print(test_dataset.class_to_idx)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
model._fc = nn.Linear(model._fc.in_features, 2)

# load the model from a checkpoint pth file
model.load_state_dict(torch.load(args.model_path))

model.to(device)
model.eval()

tp, tn, fp, fn = 0, 0, 0, 0
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
            if labels[i] == 0 and predicted[i] == 0:
                tp += 1
            elif labels[i] == 1 and predicted[i] == 1:
                tn += 1
            elif labels[i] == 1 and predicted[i] == 0:
                fp += 1
            elif labels[i] == 0 and predicted[i] == 1:
                fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}')


