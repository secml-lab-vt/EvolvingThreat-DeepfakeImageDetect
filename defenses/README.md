**Download pretrained model weights from the original repo for each of the defenses. We will provide our fine-tuned model weights soon.**

**The finetuning and inference code for all the 8 defenses on both SD and StyleCLIP datasets in our study are as follows:**

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

