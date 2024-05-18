
**Download pretrained model weights from the original repo for each of the defenses. To access the fine-tuned model weights, please fill out the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdOF6O7E-2U0q3_ISE5_NcPYg5sCFi_Q0szMf2QNrrF1HoQ-Q/viewform).**


**The finetuning and inference code for all the 8 defenses on both SD and StyleCLIP datasets in our study are as follows:**


---

# UnivCLIP

### 1. Setup
Follow steps of installation from the [original repo](https://github.com/WisconsinAIVision/UniversalFakeDetect). 

### 2. Finetuning
Pretrained model is available in `pretrained_weights`.

Finetune with the following command:
```
cd UnivCLIP
python train.py --name=<nameofexp> --wang2020_data_path=./datasets/general/ --data_mode=wang2020  --arch=CLIP:ViT-L/14  --fix_backbone --earlystop_epoch=10 --data_aug --lr=1e-3
```
* `wang2020_data_path`: Follow dataset structure similar to CNN-F defense
* `arch`: We used the CLIP:ViT-L/14 architecture
* `earlystop_epoch`: Higher number leads to more training. We found optimal performance with 10
* `lr`: 5e-4 for SD, and 1e-3 for StyleCLIP dataset


### 3. Inference
```
python test.py --arch=CLIP:ViT-L/14  --ckpt=<path to finetuned checkpoint>  --result_folder=./results/ --real_path <path to test real>  --fake_path <path to test fake> --data_mode ours --max_sample 1000
```
* `max_sample`: Our test set size was 1000 samples per class. Modify it accordingly

---


# DE-FAKE

### 1. Setup
```
cd DE-FAKE
conda env create -f environment.yaml
conda activate defake
```

### 2. Finetuning
Pretrained models can be downloaded from the [original repo](https://github.com/zeyangsha/De-Fake).

You should generate 6 csv files (train, val and test splits for real and fake data) with headers `imagepath, caption`. Imagepath should contain the full path to images, and caption should be corresponding to each images.

Finetune with the following:
```
python train.py --epoch num_epochs --lr learning_rate --inputpath_linear path --inputpath_clip path --outputpath_linear path --outputpath_clip path
```
* `epoch`: 200 for both datasets
* `lr`: 5e-4 for SD, 5e-5 for StyleCLIP dataset
* `inputpath_linear`: pretrained linear model path
* `inputpath_clip`: pretrained clip encoder model path
* `outputpath_linear`: finetuned linear model path
* `outputpath_clip`: clip encoder save path

**Note**: Fill up \<path to csv file> accordingly.

### 3. Inference
```
python test.py --outputpath_clip path --outputpath_linear path
```
**Note**: Fill up \<path to csv file> for the images you want to evaluate. Follow instructions in **Finetuning** to create the csv file.

---

# DCT
### 1. Setup
Follow steps of installation from the [original repo](https://github.com/jonasricker/diffusion-model-deepfake-detection). 


### 2. Finetuning
We do not use any pretrained weights. We train a simple linear layer network from scratch.
```
cd DCT
python train.py <path to train images> --input_size 512 --epochs 10 --lr 1e-2 --weight_decay 1e-3
```
* `input_size`: 512 for SD, 1024 for StyleCLIP


### 3. Inference
```
python test.py --fake_root <path to test fake> --real_root <path to test real> --model_path <path to trained model> --path_to_mean_std <path to mean std> --input_size 512
```
* `path_to_mean_std`: provide paths to mean and std values saved during training (from the previous step)
* `input_size`: 512 for SD, 1024 for StyleCLIP


---

# Patch-Forensics

### 1. Setup
Follow setup process in the [original repo](https://github.com/chail/patch-forensics).


### 2. Finetuning
Pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1_LekvsBFE2T9N3Wikkll3xjlogI-cSoH) and keep in a new folder `checkpoints`. We used `xception_block2` variant for our experiments. Finetune with the following command for the StyleCLIP dataset (as this defense is not applicable for the SD dataset). **Note**: The real and fake image directory structure should be similar to the one in `mydataset`.

```
cd Patch-Forensics
python3 train.py checkpoints/<checkpoint_name>/opt.yml --load_model  --which_epoch latest --overwrite_config
```

### 3. Inference
```
python3 test.py --gpu_ids 0 --which_epoch latest --partition test --dataset_name <name> --real_im_path path_to_testreal --fake_im_path path_to_testfake --train_config checkpoints/<checkpoint_name>/opt.yml
```
* `partition`: which data partition to evaluate
* `train_config`: path to finetuned checkpoint

---

# Gram-Net
### 1. Setup
```
cd Gram-Net
conda env create --name env_name --file=env.yml
conda activate env_name
```

### 2. Finetuning
Pretrained model can be downloaded from [here](https://drive.google.com/file/d/11KLxYrjRGWqXouCyi_iPgUivJKY8-7nt/view?usp=drive_link). Finetune with the following:
```
cd stylegan-ffhq
python finetune.py --model_path path_to_pretrained_model --trainlistfile <path> --val_data path_to_val_data --save_model_path <path to save finetuned model> --lr learning_rate --epoch numepochs --weight_decay decayvalue
```
* `trainlistfile`: path to a `list` file which contains <imagepath, label>. Label:0 for fake, Label:1 for real. See the provided `list` file in Gram-Net folder for reference.
* `lr`: 3e-4 for StyleCLIP, 1e-4 for SD
* `epoch`: 10 for StyleCLIP, 100 for SD
* `weight_decay`: 1e-4 for StyleCLIP, 0 for SD

### 3. Inference
```
cd stylegan-ffhq
python infer.py --fake_path path_to_fakedata --real_path path_to_realdata --model_path path_to_finetuned_checkpoint
```

---

# Resynthesis

### 1. Setup
Follow setup for Gram-Net.

### 2. Finetuning

Pretrained models can be downloaded from [here](https://drive.google.com/file/d/1FeIgABjBpjtnXT-Hl6p5a5lpZxINzXwv/view?usp=sharing) and keep in a new folder `pretrained_models`. We used `stylegan_celeba_stage5_noising` from the link. Finetune with the following command for respective datasets:

```
cd Resynthesis
bash script/finetune_styleclip.sh 0
bash script/finetune_sd.sh 0
```

Edit the following variables:
* `OUTPUT_PATH`: set path for saving models
* `DATA_ROOT_POS`: path to real training data
* `DATA_ROOT_NEG`: path to fake training data

### 3. Inference
```
python infer.py -a resnet50 --gpu 0 --data-root-pos "path_to_real_data" --data-root-neg "path_to_fake_data" --input-channel 512 --resume "/projects/secml-cs-group/sifat/Unmask/Detectors/Attack_Detectors/BeyondtheSpectrum/output/output_denoise_general/epoch_200/LR_1e_2_V1/0150.pth.tar" --sr-weights-file "/projects/secml-cs-group/sifat/Unmask/Detectors/Attack_Detectors/BeyondtheSpectrum/output/output_denoise_general/epoch_200/LR_1e_2_V1/0150_sr.pth.tar" --save_path "a_temp_path" --no_dilation --sr-scale 4 --sr-num-features 64 --sr-growth-rate 64 --sr-num-blocks 16 --sr-num-layers 8 --idx-stages 5
```
* `data-root-pos`: path to real test data
* `data-root-neg`: path to fake test data
* `resume`: path to finetuned checkpoint
* `sr-weights-file`: path to finetuned super-resolution checkpoint

---


# CNN-F

### 1. Setup
To install packages, run:

```
cd CNN-F
pip install -r requirements.txt
```

### 2. Finetuning

Download pretrained checkpoints:
```
bash weights/download_weights.sh
```

Next, finetune with the following command:

```
python train.py --name use_a_name --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot path_to_dataset --classes general --gpu_ids 0 --modeltype "0.1" --continue_train

```
* `blur_prob`: we used 0.5 for SD, and 0.1 for StyleCLIP
* `jpg_prob`: we used 0.5 for SD, and 0.1 for StyleCLIP
* `dataroot`: follow the format in `mydataset/`
* `modeltype`: "0.5" for SD, "0.1" for StyleCLIP
* `continue_train`: used for fine-tuning


### 3. Inference
```
python infer.py --dir path_to_testdata --model_path checkpointpath
```
* `model_path`: provide path to fine-tuned checkpoint file


---

# MesoNet
### 1. Setup
To install packages, follow requirements in the [original repo](https://github.com/DariusAf/MesoNet).

### 2. Finetuning
The pretrained model we use is `weights/MesoInception_DF.h5`. Finetune with the following command:
```
cd MesoNet
python finetune.py --model_path pathtomodel --dataroot pathtodata --modelsavepath path_to_save_model
```
* `model_path`: give pretrained model path
* `dataroot`: path to train and val data

We follow the same finetuning strategy for both datasets.

### 3. Inference
```
python infer.py --dir path_to_test_data --model_path path_to_finetuned_checkpoint
```

---


