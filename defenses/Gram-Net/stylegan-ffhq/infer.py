
import numpy as np
import torch,os,random
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms 
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.utils.model_zoo as model_zoo
import resnet18_gram as resnet
import os,glob
import argparse 
from torch.autograd import Variable 


parser = argparse.ArgumentParser()
parser.add_argument('--fake_path', type=str, default=None)
parser.add_argument('--real_path', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)

args = parser.parse_args()

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


root='./'

def default_loader(path):
    size = random.randint(64, 256)
    im=cv2.imread(path)
    im=cv2.resize(im,(size,size))
    im=cv2.resize(im,(512,512))
    ims=np.zeros((3,512,512))
    ims[0,:,:]=im[:,:,0]
    ims[1,:,:]=im[:,:,1]
    ims[2,:,:]=im[:,:,2]
    img_tensor=torch.tensor(ims.astype('float32'))
    
    return img_tensor

class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split(' ')[0]) for line in lines]
            self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label
 

def get_metrics(y_true, y_pred):
    tp, fp, tn, fn = 0, 0, 0, 0
    evaded_names = []
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fn += 1 
        elif y_true[i] == 1 and y_pred[i] == 0:
            fp += 1
        else:
            tn += 1 

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-10))
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    print('precision: ', precision, ' recall: ', recall, ' f1: ', f1, ' accuracy: ', accuracy) 



def test(model, args):
    model.eval()
    actuals = []
    preds = []
    valfake = args.fake_path 
    images = os.listdir(valfake)
    for i in range(len(images)): 
        im = cv2.imread(os.path.join(valfake, images[i]))
        ims = np.zeros((1, 3, 512, 512))
        ims[0, 0, :, :] = im[:, :, 0]
        ims[0, 1, :, :] = im[:, :, 1]
        ims[0, 2, :, :] = im[:, :, 2]
        image_tensor = torch.tensor(ims).float()
        inputs = Variable(image_tensor).float().cuda()
        output = model(inputs)
        output=output.detach().cpu().numpy()
        pred=np.argmax(output)
        
        actuals.append(0)
        preds.append(pred)

    valreal = args.real_path
    images = os.listdir(valreal)
    for i in range(len(images)): 
        im = cv2.imread(os.path.join(valreal, images[i]))
        ims = np.zeros((1, 3, 512, 512))
        ims[0, 0, :, :] = im[:, :, 0]
        ims[0, 1, :, :] = im[:, :, 1]
        ims[0, 2, :, :] = im[:, :, 2]
        image_tensor = torch.tensor(ims).float()
        inputs = Variable(image_tensor).float().cuda()
        output = model(inputs)
        output=output.detach().cpu().numpy()
        pred=np.argmax(output)
        
        actuals.append(1)
        preds.append(pred)

    get_metrics(actuals, preds)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet.resnet18(pretrained=True) 
resnetinit=torch.load(args.model_path)

model.load_state_dict(resnetinit.state_dict(),strict=False)
criterion = nn.NLLLoss() 
model.to(device)

test(model, args)



