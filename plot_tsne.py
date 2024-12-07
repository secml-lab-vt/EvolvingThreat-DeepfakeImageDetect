import torch
import os 
import clip
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

def seed_everything(seed=0): 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 


def plottsne(classreal, classfake, model, transform, device):
    print('transforming images') 
    classreal_tensors = torch.stack([transform(img).unsqueeze(0) for img in classreal]).squeeze(1).to(device)
    classfake_tensors = torch.stack([transform(img).unsqueeze(0) for img in classfake]).squeeze(1).to(device)

    print('encoding images')
    with torch.no_grad():
        classreal_features = model.encode_image(classreal_tensors).cpu().numpy()
        classfake_features = model.encode_image(classfake_tensors).cpu().numpy()

    print('TSNE embedding')
    all_features = np.vstack([classreal_features, classfake_features])
    embedded_features = TSNE(n_components=2).fit_transform(all_features)

    print('plotting')
    plt.figure(figsize=(10, 10))
    plt.scatter(embedded_features[:len(classreal_features), 0], embedded_features[:len(classreal_features), 1], color='g', label='REAL')
    plt.scatter(embedded_features[len(classreal_features):, 0], embedded_features[len(classreal_features):, 1], color='r', label='FAKE')
    plt.legend()
    plt.title(f"t-SNE plot")
    plt.savefig(f"tsne.png")




def main():
    seed_everything()

    device = "cuda:0" 
    model, transform = clip.load("ViT-L/14", device=device)  

    pathreal = os.path.join("<path to real images>")
    classreal = [Image.open(os.path.join(pathreal, image_name)) for image_name in os.listdir(pathreal)]

    pathfake = os.path.join("<path to fake images>")
    classfake = [Image.open(os.path.join(pathfake, image_name)) for image_name in os.listdir(pathfake)]

    plottsne(classreal, classfake, model, transform, device)
    


if __name__ == "__main__":
    main()

