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

        loss = None
        if labels is not None: 
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
          return logits, loss.item()
        else:
          return logits, None



def get_datasets():
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = torchvision.datasets.ImageFolder('<path to train>', transform=transform)
    val_ds = torchvision.datasets.ImageFolder('<path to val>', transform=transform)
    test_ds = torchvision.datasets.ImageFolder('<path to test>', transform=transform)

    return train_ds, val_ds, test_ds


def train(train_ds, val_ds, BATCH_SIZE, EPOCHS, CLASSES, LEARNING_RATE): 
    model = ViTForImageClassification(CLASSES)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()
    device = torch.device(DEVICE) 
    if torch.cuda.is_available():
        model.cuda() 

    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader  = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) 

    prev_acc = 0.0
    save_path = "<path to save finetuned model>"

    # Train the model
    for epoch in range(EPOCHS):   
        model.train()     
        for step, (x, y) in enumerate(train_loader):
            newsize = BATCH_SIZE
            if len(x) < BATCH_SIZE:
                newsize = len(x)

            x = np.split(np.squeeze(np.array(x)), newsize)
            for index, array in enumerate(x):
                x[index] = np.squeeze(array) 
            x = torch.tensor(np.stack(feature_extractor(x)['pixel_values'], axis=0))
            
            x, y  = x.to(device), y.to(device)
            b_x = Variable(x) 
            b_y = Variable(y)  
            output, temploss = model(b_x, None)
            loss = loss_func(output, b_y)    
            optimizer.zero_grad()           
            loss.backward()                 
            optimizer.step() 

            print(f'Epoch: {epoch+1}/{EPOCHS}, Step: {step+1}/{len(train_loader)}, Loss: {loss.item():.5f}')

        # Evaluate the model
        model.eval()
        accuracy = 0.0
        corrects = 0
        with torch.no_grad():
            for step, (x, y) in enumerate(val_loader): 
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
                test_output = test_output.argmax(1)

                # Calculate Accuracy
                corrects += (test_output == y).sum().item()
            accuracy = corrects / len(val_ds)
            print('Epoch: ', epoch, '| valid accuracy: %.5f' % accuracy) 
            print('Prev acc:', prev_acc, ' Curr Acc:', accuracy)
            if accuracy > prev_acc:
                prev_acc = accuracy
                torch.save(model.state_dict(), os.path.join(save_path, f"<model name>.pth"))



def main():
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)

    args = parser.parse_args()


    EPOCHS = args.epochs
    BATCH_SIZE = 64
    LEARNING_RATE = args.lr
    CLASSES = 2

    train_ds, val_ds, test_ds = get_datasets()
    train(train_ds, val_ds, BATCH_SIZE, EPOCHS, CLASSES, LEARNING_RATE) 



if __name__ == "__main__": 
    main()



