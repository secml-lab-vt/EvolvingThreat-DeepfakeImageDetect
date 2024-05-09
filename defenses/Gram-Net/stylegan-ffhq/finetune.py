
import numpy as np
import torch,os,random
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms#, models
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.utils.model_zoo as model_zoo
import resnet18_gram as resnet
import os,glob 
from torch.autograd import Variable
import argparse


parser = argparse.ArgumentParser(description='Gram-Net Finetuning')

parser.add_argument('--model_path', type=str, default=None, help='Path to the model to be loaded')
parser.add_argument('--trainlistfile', type=str, default='list_train', help='Path to the training list file')
parser.add_argument('--val_data', type=str, default=None, help='Path to the validation data directory')
parser.add_argument('--save_model_path', type=str, default='model.pth', help='Path to save the model')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

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
 
image_datasets = customData(img_path='',txt_path=(args.trainlistfile))  

train_dataloader =  torch.utils.data.DataLoader(image_datasets,
                                                 batch_size=64,
                                                 shuffle=True) 


def get_size(path):
  with open(path, 'r') as f:
    return len(f.readlines())



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet.resnet18(pretrained=True) 
resnetinit=torch.load(args.model_path)
model.load_state_dict(resnetinit.state_dict(),strict=False)


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
model.to(device)


def test6(model):
  model.eval()

  gt=0
  corr=0 
  valfake = os.path.join(args.val_data, 'fake')
  images = os.listdir(valfake)
  for i in range(len(images)): 
     im = cv2.imread(os.path.join(valfake, images[i]))
     ims = np.zeros((1, 3, 512, 512))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]
     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred=np.argmax(output) 
     if pred==0:
        corr+=1
  return corr/len(images)

def test7(model):
  model.eval()

  gt=0
  corr=0 
  valreal = os.path.join(args.val_data, 'real')
  images = os.listdir(valreal)
  for i in range(len(images)): 
     im = cv2.imread(os.path.join(valreal, images[i]))
     ims = np.zeros((1, 3, 512, 512))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]
     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred=np.argmax(output) 
     if pred==1:
        corr+=1
  return corr/len(images)


lr=args.lr
for param_group in optimizer.param_groups:
  param_group['lr']=lr


epochs = args.epoch
steps = 0
running_loss = 0
print_every = 1000
train_losses, val_losses = [], []


maxscore=0
for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in train_dataloader:
        model.train()
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        print (loss,'loss',epoch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('Epoch: {}/{}... '.format(epoch+1, epochs), "Loss: {:.4f}".format(running_loss))

    r6=test6(model)
    r7=test7(model)
    
    score = r6 + r7
    print('current score',score, ' for epoch',epoch)
    if score > maxscore:
        maxscore = score
        print('max score', maxscore, ' for epoch',epoch) 
        torch.save(model, args.save_model_path)

        

