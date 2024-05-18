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


DEVICE = "cuda:0"
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 


class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)
        return logits, None


def get_testdataset(args):
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    req_ds = torchvision.datasets.ImageFolder(args.input_path, transform=transform) 
    return req_ds


def testdata(test_ds, BATCH_SIZE, CLASSES, args):
    print("Number of test samples: ", len(test_ds))
    print("Detected Classes are: ", test_ds.class_to_idx)  

    # Define Model
    model = ViTForImageClassification(CLASSES)
    model_path = args.model_path
    model.load_state_dict(torch.load(model_path)) 

    # Feature Extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    device = torch.device(DEVICE) 
    if torch.cuda.is_available():
        model.cuda() 

    test_loader  = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) 

    model.eval()
    accuracy = 0.0
    corrects = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader): 
            newsize = BATCH_SIZE
            if len(x) < BATCH_SIZE:
                newsize = len(x)

            x = np.split(np.squeeze(np.array(x)), newsize)
            for index, array in enumerate(x):
                x[index] = np.squeeze(array)
            x = torch.tensor(np.stack(feature_extractor(x)['pixel_values'], axis=0))
            x = x.to(device)
            y = y.to(device)
            test_output, loss = model(x, y)
            pred = test_output.argmax(1)

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

    args = parser.parse_args()

    BATCH_SIZE = 64 
    CLASSES = 2

    reqds = get_testdataset(args)
    testdata(reqds, BATCH_SIZE, CLASSES, args)


if __name__ == "__main__": 
    main()
