# Zalo AI Challenge - Voice Verification
This repository contains the framework for training speaker verification model described in [2]  
with score normalization post-processing described in [3].

## Dependencies
```
pip install -r requirements.txt
```

## Data Preparation
1. Download the [public dataset](https://dl.challenge.zalo.ai/voice-verification/data/Train-Test-Data_v2.zip)
then put the training speakers data in `dataset/wavs` and public-test folder in `dataset/public-test`

2. Convert data (this will overwrite original data)
```python
python dataprep.py --save_path dataset/wavs --convert
```

3. Prepare the augment data  
```python
python dataprep.py --save_path dataset --augment
```

3. Generate train, validate list
```python
python dataprep.py --save_path dataset/wavs --generate --split_ratio -1
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

## Pretrained models
Pretrained models and corresponding cohorts can be downloaded from [here](https://drive.google.com/drive/folders/15FYmgHGKlF_JSyPGKfJzBRhQpBY5JcBw?usp=sharing).

## Training
to train model, run:
```angular2html
python main.py --do_train --augment --max_epoch 500 --batch_size 320 --initial_model checkpoints/baseline_v2_ap.model
```
## Inference
1. prepare cohorts
```angular2html
python main.py --do_infer --prepare --save_path checkpoints/cohorts_final_500_f100.npy --initial_model checkpoints/final_500.model
```
2. Evaluate and tune thresholds
```angular2html
python main.py --do_infer --eval --cohorts_path checkpoints/cohorts_final_500_f100.npy --initial_model checkpoints/final_500.model
```
3. Run on Test set
```angular2html
python main.py --do_infer --test --cohorts_path checkpoints/cohorts_final_500_f100.npy --test_threshold 1.7206447124481201 --test_path dataset --initial_model checkpoints/final_500.model
```
