from time import process_time_ns
import torch
import clip
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import itertools
import torch.nn.functional as F
import torchvision.transforms as transforms

from clipdatasets import real,fakereal
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
from torch import nn 
import argparse
import time
from tqdm import tqdm
import pandas as pd
from PIL import ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_everything(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True 


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

seed_everything()

parser = argparse.ArgumentParser(description='DEFAKE inference')
parser.add_argument('--outputpath_clip', type=str, default='output/clip.pt', help='Path to the CLIP encoder model')
parser.add_argument('--outputpath_linear', type=str, default='output/linear.pt', help='Path to the finetuned linear model')

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) 
size = 224 


class CustomRealDataset(torch.utils.data.Dataset):
    def __init__(self, datafile):  
        self.transform = transforms.Compose([
            transforms.Resize((224,224)), 
            transforms.ToTensor()
        ]) 
        self.data = pd.read_csv(datafile)  
        self.len = len(self.data.index) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx): 
        caption = self.data.iloc[idx]["caption"]
        caption = str(caption)
        label = 0

        imgpath = self.data.iloc[idx]["imagepath"]
        image = Image.open(imgpath).convert('RGB')
        image = self.transform(image)
        return image, label, caption 


class CustomFakeDataset(torch.utils.data.Dataset):
    def __init__(self, datafile):  
        self.transform = transforms.Compose([
            transforms.Resize((224,224)), 
            transforms.ToTensor()
        ]) 
        self.data = pd.read_csv(datafile)  
        self.len = len(self.data.index) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx): 
        caption = self.data.iloc[idx]["caption"]
        caption = str(caption)
        label = 1

        imgpath = self.data.iloc[idx]["imagepath"]
        image = Image.open(imgpath).convert('RGB')
        image = self.transform(image)
        return image, label, caption 



real_test = CustomRealDataset(datafile="<path to csv file>")
fake_test = CustomFakeDataset(datafile="<path to csv file>")

test_dataset = torch.utils.data.ConcatDataset([real_test, fake_test])

test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )


linear = NeuralNet(1024,[512,256],2).to(device)
linear = torch.load(args.outputpath_linear).to(device)
model = torch.load(args.outputpath_clip).to(device) 

criterion = torch.nn.CrossEntropyLoss() 

test_acc = []
test_true = []
        
for step, (x,y,t) in enumerate(tqdm(test_loader)):
    x = x.cuda()
    y = y.cuda()
    model.eval()
    linear.eval()
    text = clip.tokenize(list(t), context_length=77, truncate=True).to(device)
    with torch.no_grad():
        imga_embedding = model.encode_image(x)
        text_emb = model.encode_text(text)

    emb = torch.cat((imga_embedding,text_emb),1) 
    output = linear(emb.float())
    predict = output.argmax(1)
    predict = predict.cpu().numpy()
    predict = list(predict) 
    test_acc.extend(predict)

    y = y.cpu().numpy()
    y = list(y) 
    test_true.extend(y)   


tp, fp, tn, fn = 0, 0, 0, 0
for i in range(len(test_true)):
    if int(test_true[i]) == 1 and int(test_acc[i]) == 1:
        tp += 1
    elif int(test_true[i]) == 0 and int(test_acc[i]) == 1:
        fp += 1
    elif int(test_true[i]) == 1 and int(test_acc[i]) == 0:
        fn += 1
    elif int(test_true[i]) == 0 and int(test_acc[i]) == 0:
        tn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f'Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}') 


