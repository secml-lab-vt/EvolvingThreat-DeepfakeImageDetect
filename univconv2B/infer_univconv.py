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
import argparse
import open_clip


DEVICE = "cuda:0"
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 


class OpenCLIPForImageClassification(nn.Module):
    def __init__(self, model, num_labels=2):
        super(OpenCLIPForImageClassification, self).__init__()
        self.model = model 
        self.num_labels = num_labels 
        self.fc1 = nn.Linear(768, 1024) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_labels)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        image_features = self.model.encode_image(inputs)
        out = self.relu(self.fc1(image_features))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out 


def getdataset(path, preprocess_val):
    req_ds = torchvision.datasets.ImageFolder(path, transform=preprocess_val) 
    return req_ds


def custom_collate(batch):
    images, labels = zip(*batch) 
    return torch.stack(images), torch.tensor(labels)


def testdata(model, test_ds, BATCH_SIZE, CLASSES, args):
    print("Number of test samples: ", len(test_ds))
    print("Detected Classes are: ", test_ds.class_to_idx)  

    # Define Model
    model = OpenCLIPForImageClassification(model, CLASSES)
    model.cuda()

    model_path = args.model_path
    model.load_state_dict(torch.load(model_path))  
    
    device = torch.device(DEVICE)  
    test_loader  = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate) 

    model.eval()
    accuracy = 0.0
    corrects = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(test_loader):  
            inputs, labels = inputs.cuda(), labels.cuda() 
            logits = model(inputs) 
            
            _, pred = torch.max(logits, dim=1)
            for i in range(len(pred)):
                if pred[i] == 0 and labels[i] == 0:
                    tp += 1
                elif pred[i] == 1 and labels[i] == 1:
                    tn += 1
                elif pred[i] == 0 and labels[i] == 1:
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
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)

    args = parser.parse_args()

    BATCH_SIZE = 64 
    CLASSES = 2 
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup') 

    reqds = getdataset(args.input_path, preprocess_val)
    testdata(model, reqds, BATCH_SIZE, CLASSES, args)


if __name__ == "__main__": 
    main()
