# An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape

In this repository, we release code, datasets and model for the paper --- "[An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape](https://arxiv.org/pdf/2404.16212v1)" accepted by IEEE S&P 2024.


## Datasets and Model checkpoints
To access the datasets and model checkpoints, please fill out the [Google Form](#) (*soon to be updated*)


## Setup
```
git clone https://github.com/secml-lab-vt/EvolvingThreat-DeepfakeImageDetect.git
cd EvolvingThreat-DeepfakeImageDetect

conda env create --name env_name --file=env.yml
conda activate env_name
```


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


## Denoiser used in Sec 5.1.3
We use the [MM-BSN](https://arxiv.org/abs/2304.01598) [CVPRW 2023] denoiser. Follow instructions in [original repo](https://github.com/dannie125/MM-BSN) for installation. 

To denoise an image, run the following:
```
python test.py -c SIDD -g 0 --pretrained ./ckpt/SIDD_MMBSN_o_a45.pth --test_dir ./dataset/test_data --save_folder ./outputs/
```
* `pretrained`: path to pretrained denoiser model
* `test_dir`: path to images to be denoised
* `save_folder`: output path to save the denoised images


## Sec 5.2

### Surrogate models
```
cd surrogatemodels
```
#### Finetuning
* EfficientNet
```
python train_efficientnet.py --lr 5e-4 --epochs 30
```

* ViT
```
python train_vit.py --epochs 20 --lr 5e-5
```

* CLIP-ResNet
```
python train_clipresnet.py --lr 1e-3 --epochs 30
```


#### Inference















## Cite the paper

```
@inproceedings{abdullah2024analysis,
  title={An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape},
  author={Abdullah, Sifat Muhammad and Cheruvu, Aravind and Kanchi, Shravya and Chung, Taejoong and Gao, Peng and Jadliwala, Murtuza and Viswanath, Bimal},
  booktitle={Proc. of IEEE S\&P},
  year={2024},
}
```
