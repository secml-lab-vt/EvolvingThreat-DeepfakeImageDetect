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



def get_testdataset(args, transform):
    req_ds = torchvision.datasets.ImageFolder(args.input_path, transform=transform) 
    return req_ds

def custom_collate(batch):
    images, labels = zip(*batch) 
    return torch.stack(images), torch.tensor(labels)


def testdata(test_ds, BATCH_SIZE, CLASSES, args):
    print("Number of test samples: ", len(test_ds))
    print("Detected Classes are: ", test_ds.class_to_idx) 

    # Define Model
    model = OpenAICLIPModel(args)
    model_path = args.model_path
    model.load_state_dict(torch.load(model_path)) 

    device = torch.device(DEVICE) 
    if torch.cuda.is_available():
        model.cuda() 

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate) 

    model.eval()
    accuracy = 0.0
    corrects = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):  
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            _, pred = torch.max(logits, 1)
            corrects += torch.sum(pred == y)  

            for i in range(len(pred)):
                if pred[i] == 0 and y[i] == 0:
                    tp += 1
                elif pred[i] == 1 and y[i] == 1:
                    tn += 1
                elif pred[i] == 0 and y[i] == 1:
                    fp += 1
                else:
                    fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f"precision: {precision}, recall: {recall}, f1: {f1}, accuracy: {accuracy}")



def main():
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--modelsize', type=str, default="xxl")

    args = parser.parse_args()

    if args.modelsize=="large":
        model, preprocess = clip.load("RN50x4", device=device) # 640
    elif args.modelsize=="xl":
        model, preprocess = clip.load("RN50x16", device=device) # 768
    elif args.modelsize=="xxl":
        model, preprocess = clip.load("RN50x64", device=device)  # 1024

    BATCH_SIZE = 64 
    CLASSES = 2

    reqds = get_testdataset(args, preprocess)
    testdata(reqds, BATCH_SIZE, CLASSES, args)



if __name__ == "__main__": 
    main()
