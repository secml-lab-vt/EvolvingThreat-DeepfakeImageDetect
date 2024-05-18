import os 
import torch
import torch_dct as dct
import argparse
import random 
import numpy as np
from joblib import Parallel, delayed
import torchvision.transforms as transforms
# from sklearn.utils import shuffle
from pathlib import Path
from PIL import Image 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda:0"

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def array_from_imgdir(imgdir, grayscale=True):
    paths = []
    imgnames = os.listdir(imgdir)
    for imgname in imgnames:
        paths.append(os.path.join(imgdir, imgname))

    if grayscale:
        def loader(path):
            return transforms.ToTensor()(Image.open(path).convert("L"))
    
    array = torch.stack(
                Parallel(n_jobs=8)(delayed(loader)(path) for path in paths)
            )
    
    print('final array shape', array.shape)
    array = (array*2.0) - 1  # scale to [-1, 1]
    return array


# define a logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(LogisticRegression, self).__init__() 
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 32)
        self.linear3 = nn.Linear(32, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x): 
        out1 = self.relu(self.linear1(x))
        out2 = self.relu(self.linear2(out1))
        out3 = self.linear3(out2)
        return out3

class MyDataset(Dataset):
    def __init__(self, x_data, labels):
        self.x_data = x_data
        self.labels = labels
  
    def __len__(self):
        return len(self.x_data)
  
    def __getitem__(self, idx): 
        x = self.x_data[idx]
        y = torch.tensor(self.labels[idx])
        return x, y


def calcf1(actual, preds, args): 
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(preds)):
        if preds[i] == 1:
            if actual[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if actual[i] == 1:
                fn += 1
            else:
                tn += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print('precision', precision, ' recall', recall, ' f1', f1, ' accuracy', accuracy) 
        



def valid_epoch(model, criterion, valid_loader, args): 
    model.eval()
    
    valid_loss = 0.0
    valid_acc = 0.0
    recall_fake = 0.0
    precision_fake = 0.0
    actuals = []
    allpreds = []
    
    with torch.no_grad(): 
        for inputs, labels in valid_loader:
            outputs = model(inputs) 
            labels = labels.long()
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            valid_acc += (preds == labels).float().mean().item()
            actuals += labels.tolist()
            allpreds += preds.tolist()

    calcf1(actuals, allpreds, args)

    return valid_loss / len(valid_loader), valid_acc / len(valid_loader)



def main(args):

    fake_test = array_from_imgdir(
        args.fake_root
    )
    real_test = array_from_imgdir(
        args.real_root 
    )
    x_test = torch.cat([real_test, fake_test], dim=0)
    y_test = torch.tensor([0.0] * len(real_test) + [1.0] * len(fake_test))    
    del real_test, fake_test

    ######## Feature Calculation
    print('feature calculation...')
    x_test_tf = dct.dct_2d(x_test, norm = 'ortho')
    x_test_tf = torch.log(torch.abs(x_test_tf) + 1e-12)

    x_test_tf = x_test_tf.squeeze(1)

    x_test_tf = x_test_tf.reshape(x_test_tf.shape[0], -1)
    print('reshaped...')

    means = torch.load(os.path.join(args.path_to_mean_std, 'means.pt'))
    stds = torch.load(os.path.join(args.path_to_mean_std, 'stds.pt'))
    x_test_tf = (x_test_tf - means) / stds
    print("rescaled...")

    test_dataset = MyDataset(x_test_tf, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    #### Model Training
    print('testing model...')
    input_size = args.input_size * args.input_size 
    model = LogisticRegression(input_size, 2) 
    criterion = nn.CrossEntropyLoss()

    # load model from checkpoint
    model.load_state_dict(torch.load(args.model_path)) 
    
    test_loss, test_acc = valid_epoch(model, criterion, test_loader, args)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}') 



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fake_root",
        type=str,
        help="Root of image directory containing 'train', 'val', and test.",
    )
    parser.add_argument(
        "--real_root",
        type=str,
        help="Root of image directory containing 'train', 'val', and test.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to save model checkpoints and logs.",
    )
    parser.add_argument(
        "--path_to_mean_std",
        type=str,
        help="Path to save model checkpoints and logs.",
    ) 
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Size of the input image.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    seed_everything()

    main(parse_args()) 

