import argparse
import math
import os
import torch 
from torch import optim
from torch.nn import functional as F 
from PIL import Image
from tqdm import tqdm 
import torch.nn as nn 
import copy
from networks.resnet import resnet50
import json 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import lpips 
from model import Generator   
import random 
import numpy as np 
import clip 
from id_loss import IDLoss
import time
import pandas as pd
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from transformers import ViTFeatureExtractor
from transformers import ViTModel 
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import sys 
import torch_dct as dct
from joblib import Parallel, delayed 
import torch.optim as optim 

DEVICE = "cuda:0"

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 


def generate_wplus():
    sys.path.append("../../")
    sys.path.append("../../../")
    sys.path.append("../encoder4editing/")
    os.chdir("../encoder4editing/")

    from models.psp import pSp
    from argparse import Namespace

    dataset_name = 'ffhq'
    sys.path.append("../StyleCLIP/global_directions")
    sys.path.append("../StyleCLIP")
    os.chdir("../StyleCLIP/global_directions")

    import tensorflow as tf  
    from PIL import Image
    import pickle
    import copy
    import clip 
    import matplotlib.pyplot as plt
    from MapTS import GetFs 
    from global_directions.manipulate import Manipulator
    device = DEVICE

    M=Manipulator(dataset_name='ffhq')  
    fs3=np.load('../StyleCLIP/global_directions/npy/ffhq/fs3.npy') 
    np.set_printoptions(suppress=True) 
    experiment_type = 'ffhq_encode'
    os.chdir('../encoder4editing')

    EXPERIMENT_ARGS = {
            "model_path": "e4e_ffhq_encode.pt"
        }
    EXPERIMENT_ARGS['transform'] = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    resize_dims = (256, 256)

    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts'] 
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval() 
    net.to(DEVICE)
    print('Model successfully loaded!')
    return M, fs3, experiment_type, resize_dims, EXPERIMENT_ARGS, net


def noise_regularize(noises):
    loss = 0
    for noise in noises:
        size = noise.shape[2]
        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )
            if size <= 8:
                break
            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2
    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] 
            texts = clip.tokenize(texts).to(DEVICE)  
            class_embeddings = model.encode_text(texts) 
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(DEVICE)
    return zeroshot_weights


def GetDt(classnames, model):
    text_features=zeroshot_classifier(classnames, imagenet_templates, model).t()
    dt=text_features[0]-text_features[1]  
    dt=dt/torch.linalg.norm(dt)
    return dt 


def SplitS(ds_p,M,if_std):
    all_ds=[]
    start=0 
    for i in M.mindexs:  
        tmp=M.dlatents[i].shape[1]  
        end=start+tmp
        tmp=ds_p[start:end] 
        all_ds.append(tmp)  
        start=end
    code_std_torch = []
    for i in range(len(M.code_std)):
        code_std_torch.append(torch.from_numpy(M.code_std[i]).to(DEVICE))
    
    all_ds2=[]
    tmp_index=0
    for i in range(len(M.s_names)):
        if (not 'RGB' in M.s_names[i]) and (not len(all_ds[tmp_index])==0):
            if if_std:
                tmp=all_ds[tmp_index]*code_std_torch[i] 
            else:
                tmp=all_ds[tmp_index] 
            all_ds2.append(tmp)
            tmp_index+=1
        else:
            tmp=torch.zeros(len(M.dlatents[i][0]), requires_grad=True)
            all_ds2.append(tmp)
    return all_ds2


def GetBoundary(fs3,dt,M,threshold):
    ds_imp = torch.matmul(fs3, dt)  
    ds_imp = torch.where(torch.abs(ds_imp) < threshold, torch.tensor(0, dtype = ds_imp.dtype).to(DEVICE), ds_imp)
    tmp2 = torch.max(torch.abs(ds_imp))
    ds_imp2 = ds_imp/tmp2
    boundary_tmp2=SplitS(ds_imp2,M,if_std=True) 
    return boundary_tmp2, dt


def get_concat_h(src_image_path, im2):
    im1 = Image.open(src_image_path)
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def save_configs(configs, filename):
    with open(filename, "w") as f:
        f.write(json.dumps(configs))
    

device = DEVICE
resize = 256
M, fs3, experiment_type, resize_dims, EXPERIMENT_ARGS, net = generate_wplus()
id_loss = IDLoss() 
transform = transforms.Compose(
    [
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

def loadEffNetDetector():
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    model._fc = nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(torch.load("<path to surrogate model>"))
    model.to(DEVICE)
    model.eval()
    return model


effnet_transforms1 = transforms.Compose([
    transforms.Resize((224, 224))
])
effnet_transforms2 = transforms.Compose([ 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


criterion = nn.CrossEntropyLoss()

g_ema = Generator(1024, 512, 8)
g_ema.load_state_dict(torch.load("./checkpoint/stylegan2-ffhq-config-f.pt")["g_ema"], strict=False)
g_ema.train()
g_ema = g_ema.to(DEVICE) 

percept = lpips.PerceptualLoss(
    model="net-lin", net="vgg", use_gpu=DEVICE
)

model, preprocess = clip.load("ViT-B/32", device=DEVICE) 
fs3 = torch.from_numpy(fs3).to(DEVICE) 
import sys
import torchvision.transforms.functional as TF


if __name__ == "__main__":
    seed_everything()

    parser = argparse.ArgumentParser() 
    parser.add_argument("--savepath", type=str, default=None, help="Path to save the output images")     
    parser.add_argument("--inputpath", type=str, default=None, help="Path to input images")
    parser.add_argument("--plosscoeff", type=float, default=1.0, help="Perceptual loss coefficient")
    parser.add_argument("--classifiercoeff", type=float, default=0.1, help="Classifier loss coefficient")
    parser.add_argument("--alpha", type=float, default=9.0, help="alpha value") 
    parser.add_argument("--beta", type=float, default=0.12, help="beta value") 
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args() 

    device = DEVICE
    INPUTIMGPATH = args.inputpath 

    dfstrings = pd.read_csv("./data/neutraltargets.csv")
    allneutrals = dfstrings['neutral'].tolist()
    alltargets = dfstrings['target'].tolist()

    classifier = loadEffNetDetector()

    neutral = 'a human face'  
    target = 'a smiling face'  
    classnames=[target,neutral]
    dt_saved=GetDt(classnames,model)  

    beta = args.beta
    alpha = args.alpha
    M.alpha=[alpha]
     
    dt_saved = Parameter(dt_saved, requires_grad=True)
    dt_saved = dt_saved.to(device)

    boundary_tmp2, custom_tmp = GetBoundary(fs3,dt_saved,M,threshold=beta) 
    boundary_tmp2 = [torch.unsqueeze(tmp, 0) for tmp in boundary_tmp2]

    ORIGROOTDIR = args.savepath
    os.makedirs(ORIGROOTDIR, exist_ok=True)

    combinations = [(args.lr, "SGD", args.classifiercoeff)]

    for curcomb in combinations:
        LR, toptim, coeff = curcomb
        print('RUNNING COMBINATION:', LR, toptim, coeff)
        DESTROOT_DIR = os.path.join(ORIGROOTDIR, f"LR_{LR}_optim_{toptim}_coeff_{coeff}") 
        DEST_DIR_EVADED = os.path.join(DESTROOT_DIR, "evaded")  
        EVADED_COUNT = 0
        os.makedirs(DEST_DIR_EVADED, exist_ok=True)  

        n_mean_latent = 10000
        resize = 256
        imagelist = os.listdir(INPUTIMGPATH) 
        imgfiles = imagelist
        dt = dt_saved  

        OPTIMIZATION_ITERATION = 51
        num_gen_images = 0 
        for xind, imgfile in enumerate(imgfiles):
            rindex = random.randint(0, len(allneutrals)-1)  
            neutral = allneutrals[rindex]
            target = alltargets[rindex] 
            classnames=[target, neutral]
            dt_saved=GetDt(classnames, model) 
            dt_saved = Parameter(dt_saved, requires_grad=True)
            dt_saved = dt_saved.to(device)
            dt = dt_saved  

            g_ema.load_state_dict(torch.load("./checkpoint/stylegan2-ffhq-config-f.pt")["g_ema"], strict=False)
            g_ema.train()
            g_ema = g_ema.to(DEVICE)  

            if toptim == "SGD":
                optimizer = optim.SGD(g_ema.parameters(), lr = LR, momentum=0.9, nesterov=True)
            
            print('image name', imgfile)
            try:
                original_image = Image.open(os.path.join(INPUTIMGPATH, imgfile))
                original_image = original_image.convert("RGB") 

                def run_alignment(image_path):
                    import dlib
                    from utils.alignment import align_face
                    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                    aligned_image = align_face(filepath=image_path, predictor=predictor) 
                    print("Aligned image has shape: {}".format(aligned_image.size))
                    return aligned_image 

                if experiment_type == "ffhq_encode":
                    input_image = run_alignment(os.path.join(INPUTIMGPATH, imgfile))
                else:
                    input_image = original_image

                input_image.resize(resize_dims)

                img_transforms = EXPERIMENT_ARGS['transform']
                transformed_image = img_transforms(input_image)

                def run_on_batch(inputs, net):
                    images, latents = net(inputs.to(DEVICE).float(), randomize_noise=False, return_latents=True)
                    if experiment_type == 'cars_encode':
                        images = images[:, :, 32:224, :]
                    return images, latents

                with torch.no_grad():
                    images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
                    result_image, latent = images[0], latents[0]

            except Exception as e:
                print(f"An exception occured! {e}")  
                continue

            img = transform(Image.open(os.path.join(INPUTIMGPATH, imgfile)).convert("RGB")) 
            imgs = []
            imgs.append(img) 
            imgs = torch.stack(imgs, 0).to(device) 
            
            with torch.no_grad():
                noise_sample = torch.randn(n_mean_latent, 512, device=device)  
                latent_out = g_ema.style(noise_sample)  

                latent_mean = latent_out.mean(0)
                latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

            noises_single = g_ema.make_noise()
            noises = []
            for noise in noises_single:
                noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1) 

            if True:
                latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

            latent_in.requires_grad = True 

            for noise in noises:
                noise.requires_grad = True
            
            flag = 'Y'
            if(flag =='Y'):
                latent_out = latents
                latent_in = latent_out.to(device)
                latent_in.requires_grad = True
            
            latent_path = []
            i = 0
            img_gen = None
            latent_path.append(latent_in.detach().clone())  

            for tempi in range(OPTIMIZATION_ITERATION): 
                try:
                    boundary_tmp2, custom_tmp = GetBoundary(fs3,dt,M,threshold=beta) 
                    boundary_tmp2 = [torch.unsqueeze(tmp, 0) for tmp in boundary_tmp2]
                    img_gen, _ = g_ema(latent_path[-1], input_is_latent=True, noise=noises,flag=flag, boundary_tmp2=boundary_tmp2, alpha=alpha, use_dt=True)  

                    batch, channel, height, width = img_gen.shape 
                    if height > 256:
                        factor = height // 256
                        img_gen2 = img_gen.reshape(
                            batch, channel, height // factor, factor, width // factor, factor
                        )
                        img_gen3 = img_gen2.mean([3, 5])   

                    ten0 = effnet_transforms1(img_gen) 
                    tensor1 = ten0.clamp_(min=-1, max=1) 
                    tensor2 = tensor1.add(1) 
                    tensor3 = torch.squeeze(tensor2.div_(2))
                    tensor4 = effnet_transforms2(tensor3)
                    tensor4 = tensor4.unsqueeze(0)
                    tensor5 = tensor4.cuda() 
                    output = classifier(tensor5)  

                    targetlabel = torch.tensor([1.0], dtype=torch.long).to(DEVICE) 
                    
                    classifier_loss = criterion(output, targetlabel)
                    classifier_loss_value = classifier_loss.item()  
                    p_loss = percept(img_gen3, imgs, listofimages=False).sum()

                    
                    loss = args.plosscoeff * p_loss + coeff * classifier_loss 

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()  
                    torch.autograd.set_detect_anomaly(True)

                    if output[0][0] < output[0][1] and tempi == OPTIMIZATION_ITERATION-1:
                        img_ar = make_image(img_gen) 
                        tempdir = os.path.join(DEST_DIR_EVADED, f"iter_{tempi}") 
                        os.makedirs(tempdir, exist_ok=True) 
                        img_name = os.path.join(tempdir, f"img_{tempi}_{imgfile}") 
                        pil_img = Image.fromarray(img_ar[0])
                        pil_img.save(img_name) 

                except Exception as e:
                    print(f"An exception occured! {e}")  
                    continue

            torch.cuda.empty_cache()
