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
    imgnames = np.sort(imgnames)
    for imgname in imgnames:
        paths.append(os.path.join(imgdir, imgname))

    if grayscale:
        def loader(path):
            x = transforms.ToTensor()(Image.open(path).convert("L")) 
            return x 
    
    array = torch.stack([loader(path) for path in paths])
    
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
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x): 
        out1 = self.relu(self.linear1(x))
        out1 = self.dropout1(out1)
        out2 = self.relu(self.linear2(out1))
        out2 = self.dropout2(out2)
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
    os.makedirs(args.output_path, exist_ok=True)

    real_train = array_from_imgdir(
        os.path.join(args.image_root, "train", "real")
    ) 
    fake_train = array_from_imgdir(
        os.path.join(args.image_root, "train", "fake")
    )

    noise_real_train = array_from_imgdir(
        os.path.join(args.noise_image_root, "train", "real")
    ) 
    noise_fake_train = array_from_imgdir(
        os.path.join(args.noise_image_root, "train", "fake")
    )

    x_train = torch.cat([real_train, fake_train], dim=0)
    y_train = torch.tensor([0.0] * len(real_train) + [1.0] * len(fake_train)) 
    noise_x_train = torch.cat([noise_real_train, noise_fake_train], dim=0)
    del real_train, fake_train, noise_real_train, noise_fake_train

    real_val = array_from_imgdir(
        os.path.join(args.image_root, "val", "real")
    )
    fake_val = array_from_imgdir(
        os.path.join(args.image_root, "val", "fake")
    )

    noise_real_val = array_from_imgdir(
        os.path.join(args.noise_image_root, "val", "real")
    )
    noise_fake_val = array_from_imgdir(
        os.path.join(args.noise_image_root, "val", "fake")
    )
    x_val = torch.cat([real_val, fake_val], dim=0)
    y_val = torch.tensor([0.0] * len(real_val) + [1.0] * len(fake_val))
    noise_x_val = torch.cat([noise_real_val, noise_fake_val], dim=0)
    del real_val, fake_val, noise_real_val, noise_fake_val

    x_train_tf = dct.dct_2d(x_train, norm = 'ortho')
    x_train_tf = torch.log(torch.abs(x_train_tf) + 1e-12)
    x_val_tf = dct.dct_2d(x_val, norm = 'ortho')
    x_val_tf = torch.log(torch.abs(x_val_tf) + 1e-12)

    noise_x_train_tf = dct.dct_2d(noise_x_train, norm = 'ortho')
    noise_x_train_tf = torch.log(torch.abs(noise_x_train_tf) + 1e-12)
    noise_x_val_tf = dct.dct_2d(noise_x_val, norm = 'ortho')
    noise_x_val_tf = torch.log(torch.abs(noise_x_val_tf) + 1e-12)

    x_train_tf = x_train_tf.squeeze(1) 
    x_val_tf = x_val_tf.squeeze(1) 
    noise_x_train_tf = noise_x_train_tf.squeeze(1)
    noise_x_val_tf = noise_x_val_tf.squeeze(1)

    x_train_tf = x_train_tf.reshape(x_train_tf.shape[0], -1) 
    x_val_tf = x_val_tf.reshape(x_val_tf.shape[0], -1) 
    noise_x_train_tf = noise_x_train_tf.reshape(noise_x_train_tf.shape[0], -1)
    noise_x_val_tf = noise_x_val_tf.reshape(noise_x_val_tf.shape[0], -1)

    means = x_train_tf.mean(0, keepdim=True)
    stds = x_train_tf.std(0, unbiased=False, keepdim=True)

    noise_means = noise_x_train_tf.mean(0, keepdim=True)
    noise_stds = noise_x_train_tf.std(0, unbiased=False, keepdim=True)

    torch.save(means, os.path.join(args.output_path, 'means.pt'))
    torch.save(stds, os.path.join(args.output_path, 'stds.pt'))

    torch.save(noise_means, os.path.join(args.output_path, 'noise_means.pt'))
    torch.save(noise_stds, os.path.join(args.output_path, 'noise_stds.pt'))

    x_train_tf = (x_train_tf - means) / stds 
    x_val_tf = (x_val_tf - means) / stds  

    noise_x_train_tf = (noise_x_train_tf - noise_means) / noise_stds
    noise_x_val_tf = (noise_x_val_tf - noise_means) / noise_stds

    combined_x_train = torch.cat([x_train_tf, noise_x_train_tf], dim=1)
    combined_x_val = torch.cat([x_val_tf, noise_x_val_tf], dim=1)

    print('shape of combined_x_train', combined_x_train.shape)
    print('shape of combined_x_val', combined_x_val.shape) 

    train_dataset = MyDataset(combined_x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = MyDataset(combined_x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True) 

    print('training model...') 
    input_size = 512 * 512 * 2 
    model = LogisticRegression(input_size, num_classes=2) 
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 30
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
            torch.save(model.state_dict(), os.path.join(args.output_path, f'<name of trained model>.pth'))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_root",
        type=str,
        help="Root of image directory containing 'train', 'val', and test.",
        default=None 
    )
    parser.add_argument(
        "--noise_image_root",
        type=str,
        help="Root of image directory containing 'train', 'val', and test.",
        default=None 
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Root of image directory containing 'train', 'val', and test.",
        default=None 
    )

    return parser.parse_args()


if __name__ == "__main__":
    seed_everything()

    main(parse_args())
