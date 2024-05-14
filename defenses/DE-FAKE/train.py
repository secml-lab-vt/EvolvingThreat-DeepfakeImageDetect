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
ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser(description='DE-FAKE finetuning')
parser.add_argument('--epoch', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--inputpath_linear', type=str, default=None, help='path to pretrained linear model')
parser.add_argument('--inputpath_clip', type=str, default=None, help='path to pretrained CLIP model')
parser.add_argument('--outputpath_linear', type=str, default=None, help='path to save linear model - should be like filename.pt')
parser.add_argument('--outputpath_clip', type=str, default=None, help='path to save CLIP model - should be like filename.pt')

args = parser.parse_args()


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


real_train = CustomRealDataset(datafile="<path to csv file>")
real_val = CustomRealDataset(datafile="<path to csv file>")
real_test = CustomRealDataset(datafile="<path to csv file>")

fake_train = CustomFakeDataset(datafile="<path to csv file>")
fake_val = CustomFakeDataset(datafile="<path to csv file>")
fake_test = CustomFakeDataset(datafile="<path to csv file>")

train_dataset = torch.utils.data.ConcatDataset([real_train, fake_train])
val_dataset = torch.utils.data.ConcatDataset([real_val, fake_val])
test_dataset = torch.utils.data.ConcatDataset([real_test, fake_test])

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )

val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )

test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )



linear = NeuralNet(1024,[512,256],2).to(device)


linear = torch.load(args.inputpath_linear).to(device)
model = torch.load(args.inputpath_clip).to(device)

model.eval()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(linear.parameters())+list(model.parameters()), lr=args.lr)

for i in range(args.epoch): 
    loss_epoch = 0
    train_acc = []
    train_true = []
    
    test_acc = []
    test_true = []

    for step, (x,y,t) in enumerate(tqdm(train_loader)):
        x = x.cuda()
        y = y.cuda()
        linear.train()  
        text = clip.tokenize(list(t), context_length=77, truncate=True).to(device)
        with torch.no_grad():
            imga_embedding = model.encode_image(x)
            text_emb = model.encode_text(text)
        emb = torch.cat((imga_embedding,text_emb),1)
        output = linear(emb.float())
        optimizer.zero_grad()
        loss = criterion(output,y)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        predict = output.argmax(1)
        predict = predict.cpu().numpy()
        predict = list(predict)
        train_acc.extend(predict)
        
        y = y.cpu().numpy()
        y = list(y)
        train_true.extend(y)
        
    for step, (x,y,t) in enumerate(tqdm(val_loader)):
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
    
    print('train')
    print(accuracy_score(train_true,train_acc)) 
    print('validation')
    print(accuracy_score(test_true,test_acc)) 

    val_acc = accuracy_score(test_true,test_acc)

    # save model
    torch.save(linear, args.outputpath_linear)
    torch.save(model, args.outputpath_clip)



