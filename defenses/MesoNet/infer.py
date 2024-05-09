import numpy as np
import os
from classifiers import *
import pandas as pd
import random 
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint 
import tensorflow
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.preprocessing import image
from tqdm import tqdm
import scipy
import shutil
import argparse 


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)


def confusion_matrix(y_true, y_preds):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(y_true.size):
        if y_true[i]==0 and y_preds[i]==0:
            tp += 1
        elif y_true[i]==1 and y_preds[i]==1:
            tn += 1 
        elif y_true[i]==1 and y_preds[i]==0:
            fp += 1
        elif y_true[i]==0 and y_preds[i]==1:
            fn += 1 

    recall = tp/(tp+fn + 1e-10)
    precision = tp/(tp+fp + 1e-10)
    f1_score = 2*(recall*precision)/(recall+precision + 1e-10)
    accuracy = (tp+tn)/(tp+tn+fp+fn + 1e-10)
    print('precision: ', precision, ' recall: ', recall, ' f1_score: ', f1_score, ' accuracy: ', accuracy)




def test(args):
    # Testing
    classifier = MesoInception4()
    classifier.load(args.model_path)
    dataGenerator = ImageDataGenerator(rescale=1. / 255)

    generator = dataGenerator.flow_from_directory(
        args.dir,
        target_size=(256, 256),
        batch_size=32,
        shuffle=False,
        class_mode='binary'
        ) 

    y_preds = classifier.model.predict(
        generator,
        steps=len(generator),
        verbose=1
    )
    y_preds = np.where(np.squeeze(y_preds) >= 0.5, 1, 0)
    y_true = generator.classes
    
    confusion_matrix(y_true, y_preds)
    



def main():
    seed_everything()

    parser = argparse.ArgumentParser(description='MesoNet Testing')

    parser.add_argument('--dir', type=str, default=None) 
    parser.add_argument('--model_path', type=str, default=None)

    args = parser.parse_args()
    
    
    test(args)


if __name__ == "__main__":
    main()




