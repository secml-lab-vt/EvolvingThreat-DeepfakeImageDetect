# An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape

In this repository, we release code, datasets and model checkpoints for the paper --- "[An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape](https://arxiv.org/pdf/2404.16212v1)" accepted by IEEE S&P 2024.


## Datasets and Model checkpoints
To access the datasets and model checkpoints, please fill out the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdOF6O7E-2U0q3_ISE5_NcPYg5sCFi_Q0szMf2QNrrF1HoQ-Q/viewform) 

---

## Setup
```
git clone https://github.com/secml-lab-vt/EvolvingThreat-DeepfakeImageDetect.git
cd EvolvingThreat-DeepfakeImageDetect

conda env create --name env_name --file=env.yml
conda activate env_name
```

---

## Defenses we studied

Installation, finetuning and inference instructions for the 8 defenses that we have studied are in `defenses` folder. Please follow the `README` file in `defenses`.

---



## Sec 5.1.3 : Content-agnostic features
### Denoiser
We use the [MM-BSN](https://arxiv.org/abs/2304.01598) [CVPRW 2023] denoiser to get denoised images. Follow instructions in [original repo](https://github.com/dannie125/MM-BSN) for installation. 

To denoise an image, run the following:
```
python test.py -c SIDD -g 0 --pretrained ./ckpt/SIDD_MMBSN_o_a45.pth --test_dir ./dataset/test_data --save_folder ./outputs/
```
* `pretrained`: path to pretrained denoiser model
* `test_dir`: path to images to be denoised
* `save_folder`: output path to save the denoised images

### Extract Noise
```
cd contentagnostic
python extractnoise.py --origpath <path to original images> --denpath <path to denoised images> --outputpath <path where image noise will be saved>
```

### Train DCT + Noise features
```
python traindct_w_noise.py --image_root <path to train images> --noise_image_root <path to noise of train images> --output_path <path to save trained model>
```

### Inference
```
python testdct_w_noise.py --fake_root <path to test fake> --real_root <path to test real> --noise_fake_root <path to noise of test fake images> --noise_real_root <path to noise of test real images> --model_path <path to trained model> --path_to_mean_std <path to saved mean and std values during training>
```

---



## Sec 5.2 : Our Adversarial Attack

### Finetuning Surrogate Models
```
cd surrogatemodels
```

For finetuning CLIP-ResNet:
```
python train_clipresnet.py --lr 1e-3 --epochs 30
```
We provide scripts in `surrogatemodels` for EfficientNet and ViT finetuning, similar to CLIP-ResNet.

CLIP-ResNet inference:

```
python infer_clipresnet.py --model_path <path to finetuned model> --input_path <path to test data>
```
We provide scripts in `surrogatemodels` for EfficientNet and ViT inference, similar to CLIP-ResNet.


### Script for adversarial attack
```
cd adversarialattack/stylegan2-pytorch
```
Run our adversarial attack for the CLIP-ResNet surrogate classifier with the following command:
```
python adversarialattack_clipresnet.py --inputpath ./dataset/ --savepath ./outputs/ --plosscoeff 1.0 --classifiercoeff 0.1 --alpha 9.0 --beta 0.12 --lr 1e-3
```
* `inputpath`: path to input images for adversarial manipulations
* `savepath`: path to save adversarial images
* `plosscoeff`: perceptual loss coefficient. We use 1.0 always
* `classifiercoeff`: classifier loss coefficient. We use 0.1 for EfficientNet and ViT, and 0.02 for CLIP-ResNet
* `lr`: we use learning rate of 1e-3

**Provide finetuned surrogate classifier path accordingly in the script**. We also provide similar scripts in `adversarialattack/stylegan2-pytorch` for running adversarial attack with EfficientNet and ViT surrogate deepfake classifiers. 




### UnivConv2B defense
```
cd univconv2B
```
#### Finetuning
```
python train_univconv.py --epochs 30 --lr 1e-3
```

#### Inference
```
python infer_univconv.py --model_path <path to finetuned model> --input_path <path to test data>
```

---



## Metrics

### KID

#### Install library
```
pip install clean-fid
```

#### Calculate KID value
```
cd Metrics
python calcKID.py --dir1 <path to first directory of images> --dir2 <path to second directory of images>
```
Basically, provide paths to image directories you want to calculate KID for.


### CLIP-Score
We followed the instructions from the [original repo](https://github.com/jmhessel/clipscore). 



---



## Cite the paper

```
@inproceedings{abdullah2024analysis,
  title={An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape},
  author={Abdullah, Sifat Muhammad and Cheruvu, Aravind and Kanchi, Shravya and Chung, Taejoong and Gao, Peng and Jadliwala, Murtuza and Viswanath, Bimal},
  booktitle={Proc. of IEEE S\&P},
  year={2024},
}
```

