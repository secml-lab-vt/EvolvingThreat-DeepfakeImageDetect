import numpy as np
import os
from classifiers import *
import pandas as pd
import random
# from pipeline import *
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
# from keract import get_activations
# from keract import display_heatmaps
# from PIL import Image
# from tensorflow import set_random_seed
import tensorflow
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import wandb
from wandb.keras import WandbCallback
from keras.preprocessing import image
from tqdm import tqdm
import scipy 
import matplotlib.pyplot as plt
import shutil
import argparse
# tensorflow.compat.v1.disable_eager_execution()
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)



def train(args):
    # Training
    classifier = MesoInception4()
    # load weights into new model
    classifier.model.load_weights(args.model_path)

    # Freeze the layers except the last 2 layers
    # for layer in classifier.model.layers[:-2]:
    #     layer.trainable = False


    dataGenerator = ImageDataGenerator(rescale=1. / 255)

    train_generator = dataGenerator.flow_from_directory(os.path.join(args.dataroot, "train"), 
        target_size=(256, 256),
        batch_size=64,
        class_mode='binary')

    valid_generator = dataGenerator.flow_from_directory(os.path.join(args.dataroot, "val"), 
        target_size=(256, 256),
        batch_size=64,
        class_mode='binary')

    checkpoint = ModelCheckpoint(os.path.join(args.modelsavepath, "modeloutput.h5"), 
        monitor="val_accuracy", mode="max",
        save_best_only=True,
        verbose=1)


    history = classifier.model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // 64,
        validation_data=valid_generator,
        validation_steps=valid_generator.n//64,
        verbose=1,
        epochs=100,
        workers=8,
        callbacks=[checkpoint]
    )

    print(max(history.history["accuracy"]))
    print(min(history.history["loss"]))
    print(max(history.history["val_accuracy"]))
    print(min(history.history["val_loss"]))




def main():
    seed_everything()

    parser = argparse.ArgumentParser(description='MesoNet Finetuning')
    parser.add_argument('--model_path', type=str, help='path to pretrained checkpoint')
    parser.add_argument('--dataroot', type=str, help='path to dataset for finetuning')
    parser.add_argument('--modelsavepath', type=str, help='path to save the model')

    args = parser.parse_args()

    
    train(args)



if __name__ == "__main__":
    main()


