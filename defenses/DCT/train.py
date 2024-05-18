import os 
import torch
import torch_dct as dct
import argparse
import random 
import numpy as np
from joblib import Parallel, delayed
import torchvision.transforms as transforms 
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


def train_epoch(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs) 
        labels = labels.long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_acc += (preds == labels).float().mean().item()
    
    return train_loss / len(train_loader), train_acc / len(train_loader)


def valid_epoch(model, criterion, valid_loader):
    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad(): 
        for inputs, labels in valid_loader:
            outputs = model(inputs) 
            labels = labels.long()
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            valid_acc += (preds == labels).float().mean().item()
    
    return valid_loss / len(valid_loader), valid_acc / len(valid_loader)



def main(args):

    real_train = array_from_imgdir(
        args.image_root / "train" / "real"
    )
    fake_train = array_from_imgdir(
        args.image_root / "train" / "fake"
    )
    x_train = torch.cat([real_train, fake_train], dim=0)
    y_train = torch.tensor([0.0] * len(real_train) + [1.0] * len(fake_train))
    del real_train, fake_train

    real_val = array_from_imgdir(
        args.image_root / "val" / "real" 
    )
    fake_val = array_from_imgdir(
        args.image_root / "val" / "fake" 
    )
    x_val = torch.cat([real_val, fake_val], dim=0)
    y_val = torch.tensor([0.0] * len(real_val) + [1.0] * len(fake_val))
    del real_val, fake_val

    print('feature calculation...')
    x_train_tf = dct.dct_2d(x_train, norm = 'ortho')
    x_train_tf = torch.log(torch.abs(x_train_tf) + 1e-12)

    x_val_tf = dct.dct_2d(x_val, norm = 'ortho')
    x_val_tf = torch.log(torch.abs(x_val_tf) + 1e-12)

    x_train_tf = x_train_tf.squeeze(1)
    x_val_tf = x_val_tf.squeeze(1)
    

    x_train_tf = x_train_tf.reshape(x_train_tf.shape[0], -1)
    x_val_tf = x_val_tf.reshape(x_val_tf.shape[0], -1)
    print('reshaped...')

    means = x_train_tf.mean(0, keepdim=True)
    stds = x_train_tf.std(0, unbiased=False, keepdim=True)

    torch.save(means, '<path to save>/means.pt')
    torch.save(stds, '<path to save>/stds.pt')

    x_train_tf = (x_train_tf - means) / stds
    x_val_tf = (x_val_tf - means) / stds
    print("rescaled...")
    
    train_dataset = MyDataset(x_train_tf, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = MyDataset(x_val_tf, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    #### Model Training
    print('training model...')
    input_size = args.input_size * args.input_size
    model = LogisticRegression(input_size, num_classes=2) 
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    criterion = nn.CrossEntropyLoss()

    num_epochs = args.epochs 
    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader)
        valid_loss, valid_acc = valid_epoch(model, criterion, val_loader)
        
        # Print the epoch statistics
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}')
        
        # Save the model with the best validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f'<path to save model>.pth')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_root",
        type=Path,
        help="Root of image directory containing 'train', 'val', and test.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Size of input image",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay regularization value",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    seed_everything()

    main(parse_args())
