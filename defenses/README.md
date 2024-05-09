
**Download pretrained model weights from the original repo for each of the defenses. We will provide our fine-tuned model weights soon.**

**The fine-tuning and inference code for all the 8 defenses on both SD and StyleCLIP datasets in our study are as follows:**

---

# CNN-F

### 1. Setup
To install packages, run:

```
cd CNN-F
pip install -r requirements.txt
```

### 2. Fine-tuning

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

### 2. Fine-tuning
The pretrained model we use is `weights/MesoInception_DF.h5`. Finetune with the following command:
```
cd MesoNet
python finetune.py --model_path pathtomodel --dataroot pathtodata --modelsavepath path_to_save_model
```
* `model_path`: give pretrained model path
* `dataroot`: path to train and val data

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

### 2. Fine-tuning
Pretrained model can be downloaded from [here](https://drive.google.com/file/d/11KLxYrjRGWqXouCyi_iPgUivJKY8-7nt/view?usp=drive_link). Fine-tune with the following:
```
python finetune.py --model_path path_to_pretrained_model --trainlistfile <path> --val_data path_to_val_data --save_model_path <path to save finetuned model>
```
* `trainlistfile`: path to a `list` file which contains <imagepath, label>. Label:0 for fake, Label:1 for real. See the provided `list` file in Gram-Net folder for reference.

### 3. Inference
```
python infer.py --fake_path path_to_fakedata --real_path path_to_realdata --model_path path_to_finetuned_checkpoint
```

---

