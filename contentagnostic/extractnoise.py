
import cv2
import os 
from tqdm import tqdm
import argparse

def subtract_images(origpath, denpath, outputpath):
    origimages = os.listdir(origpath)
    denimages = os.listdir(denpath)

    for image in tqdm(origimages):
        onlyname = image.split('.')[0]
        dimage = onlyname + '_DN.png'

        oimgpath = os.path.join(origpath, image)
        dimgpath = os.path.join(denpath, dimage)

        img1 = cv2.imread(oimgpath)
        img2 = cv2.imread(dimgpath)

        absdiff = cv2.absdiff(img1, img2) 
        absname = onlyname + '_abs.png' 
        absoutpath = os.path.join(outputpath, absname) 
        cv2.imwrite(absoutpath, absdiff) 



parser = argparse.ArgumentParser(description='Extract noise from images')
parser.add_argument('--origpath', type=str, help='Path to original images')
parser.add_argument('--denpath', type=str, help='Path to denoised images')
parser.add_argument('--outputpath', type=str, help='Path to save image noises')

args = parser.parse_args()

os.makedirs(args.outputpath, exist_ok=True) 

subtract_images(args.origpath, args.denpath, args.outputpath)
