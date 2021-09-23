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
   or from this [link](https://drive.google.com/drive/folders/1b_Ded7l_59IxIBz4H6Ok5l1knbNjvj04?usp=sharing)
2. Convert data (this will overwrite original data) to have all 16kHz

```python
python dataprep.py  --convert
```

3. Prepare the augment data

```python
python dataprep.py --augment --aug_rate -1
```

3. Generate train, validate list

```python
python dataprep.py --generate --split_ratio -1
```

4. Transform data to npy format (optional)

```python
python dataprep.py --transform
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

## Pretrained models

Pretrained models and corresponding cohorts can be downloaded from [here](https://drive.google.com/drive/folders/15FYmgHGKlF_JSyPGKfJzBRhQpBY5JcBw?usp=sharing).

## Training

**Phase 1**: Train with classification loss (softmax, amsoftmax, aamsoftmax)

```python
!python main.py --do_train \
                --train_list dataset/train.txt \
                --test_list dataset/val.txt \
                --model ResNetSE34V2 \
                --max_epoch 500 \
                --batch_size 128 \
                --nDataLoaderThread 2 \
                --trainfunc amsoftmax \
                --margin 0.1\
                --scale 30\
                --nPerSpeaker 1
                --initial_model checkpoints/baseline_v2_ap.model
```

**Phase 2**: Train with metric loss (angle, proto, angleproto, triplet, metric)

```python
!python main.py --do_train \
                --train_list dataset/train.txt \
                --test_list dataset/val.txt \
                --model ResNetSE34V2 \
                --max_epoch 600 \
                --batch_size 128 \
                --nDataLoaderThread 2 \
                --trainfunc angleproto \
                --nPerSpeaker 2
```

**Or**, train with combined loss(softmaxproto, amsoftmaxproto)

```python
!python main.py --do_train \
                --train_list dataset/train.txt \
                --test_list dataset/val.txt \
                --model ResNetSE34V2 \
                --max_epoch 500 \
                --batch_size 128 \
                --nDataLoaderThread 2 \
                --trainfunc softmaxproto \
                --nPerSpeaker 2 \
                --initial_model checkpoints/baseline_v2_ap.model
```

Note: the best model is automaticly saved during the training process, if the initial_model is not provided, automaticly load from the best_state weight if possible.

## Inference

1. prepare cohorts

```python
!python main.py --do_infer --prepare \
                --model ResNetSE34V2 \
                --test_list dataset/val.txt \
                --cohorts_path checkpoints/cohorts_resnet34v2.npy \
                --initial_model_infer exp/ResNetSE34v2/model/best_state.model
```

2. Evaluate and tune thresholds

```python
!python main.py --do_infer --eval \
                --model ResNetSE34V2 \
                --test_list dataset/val.txt \
                --cohorts_path checkpoints/cohorts_resnet34v2.npy \
                --initial_model_infer exp/ResNetSE34v2/model/best_state.model
```

3. Run on Test set

```python
!python main.py --do_infer --test \
                --model ResNetSE34V2 \
                --cohorts_path checkpoints/cohorts_resnet34v2.npy \
                --test_threshold 1.7206447124481201 \
                --test_path dataset \
                --initial_model_infer exp/ResNetSE34v2/model/best_state.model
```

4. test each pair(to get the predict time of each pair):

```python
!python main.py --do_infer --test_by_pair \
                --model ResNetSE34V2 \
                --cohorts_path checkpoints/cohorts_resnet34v2.npy \
                --test_threshold 1.7206447124481201 \
                --test_path dataset \
                --initial_model_infer exp/ResNetSE34v2/model/best_state.model
```
