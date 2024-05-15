import os 
from transformers import ViTFeatureExtractor 
import torchvision
from torchvision.transforms import ToTensor
from torchvision import transforms
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor
import torch.nn as nn
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import random
import shutil 
from PIL import Image
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import argparse
import clip


DEVICE = "cuda:0"
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 



class OpenAICLIPModel(nn.Module):
    def __init__(self, args, num_labels=2):
        super(OpenAICLIPModel, self).__init__()
        if args.modelsize=="large":
            self.model, self.preprocess = clip.load("RN50x4", device="cuda") # 640
        elif args.modelsize=="xl":
            self.model, self.preprocess = clip.load("RN50x16", device="cuda")
        elif args.modelsize=="xxl":
            self.model, self.preprocess = clip.load("RN50x64", device="cuda")
            
        self.num_labels = num_labels 

        if args.modelsize=="large":
            self.fc1 = nn.Linear(640, 1024)  
        elif args.modelsize=="xl":
            self.fc1 = nn.Linear(768, 1024)
        elif args.modelsize=="xxl":
            self.fc1 = nn.Linear(1024, 1024)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_labels)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        image_features = self.model.encode_image(inputs)
        image_features = image_features.float()
        out = self.relu(self.fc1(image_features)) 
        out = self.relu(self.fc2(image_features)) 
        out = self.fc3(out)
        return out 


def custom_collate(batch):
    images, labels = zip(*batch) 
    return torch.stack(images), torch.tensor(labels)


def train(train_ds, val_ds, EPOCHS, BATCH_SIZE, CLASSES, LEARNING_RATE, args): 
    device = torch.device(DEVICE)   
    model = OpenAICLIPModel(args)
    model.cuda()

    params = [] 
    for name, p in model.named_parameters():
        if  name=="fc1.weight" or name=="fc1.bias" or name=="fc2.weight" or name=="fc2.bias" or name=="fc3.weight" or name=="fc3.bias": 
            params.append(p) 
        else:
            p.requires_grad = False
            
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader  = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate) 

    num_epochs = EPOCHS 

    prev_acc = 0.0 

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader): 
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad() 
            logits = model(inputs)
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 1 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda() 
                logits = model(inputs)
                
                loss = loss_func(logits, labels)
                total_val_loss += loss.item()
                
                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions.double() / len(val_ds)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > prev_acc:
            prev_acc = val_accuracy
            torch.save(model.state_dict(), os.path.join(args.savepath, f"<finetuned model name>.pth"))



def get_datasets(preprocess):

    train_ds = torchvision.datasets.ImageFolder('<path to train data>', transform=preprocess)
    val_ds = torchvision.datasets.ImageFolder('<path to val data>', transform=preprocess) 

    return train_ds, val_ds 



def main():
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--savepath", type=str, default="./output_checkpoints")  
    parser.add_argument("--modelsize", type=str, default="xxl")
    parser.add_argument("--epochs", type=int, default=30)

    args = parser.parse_args()
    os.makedirs(args.savepath, exist_ok=True) 

    device = "cuda" 
    if args.modelsize=="large":
        model, preprocess = clip.load("RN50x4", device=device) # 640
    elif args.modelsize=="xl":
        model, preprocess = clip.load("RN50x16", device=device) # 768
    elif args.modelsize=="xxl":
        model, preprocess = clip.load("RN50x64", device=device)  # 1024
 
    EPOCHS = args.epochs
    BATCH_SIZE = 64
    LEARNING_RATE = args.lr 
    CLASSES = 2

    print('loading datasets')
    train_ds, val_ds = get_datasets(preprocess)

    train(train_ds, val_ds, EPOCHS, BATCH_SIZE, CLASSES, LEARNING_RATE, args)



if __name__ == "__main__": 
    main()



