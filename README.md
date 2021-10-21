# Voice Verification for Zalo AI Challenge Dataset

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
   
   Structure of dataset:
   dataset ---- wavs ---- id_00001 ---- id_00001_1.wav
                    |             |---- id_00001_1.wav
                    |             |---- id_00001_1.wav
                    |             \----  ....
                    |---- id_00002 ---- ....
                    \---- id_....
                    
2. Convert data (this will overwrite original data) to have all 16kHz

```python
python dataprep.py  --convert
```

3. Prepare the augment data

```python
python dataprep.py --augment --aug_rate -1
```

3. Generate train, validate list
   (if ratio == -1, take 3 files for each speaker for validate)

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

**Set cuda usage**

```
!export CUDA_VISIBLE_DEVICES=5
```

then add the device="cuda:5" to args
**Phase 1**: Train with classification loss (softmax, amsoftmax, aamsoftmax)

```python
!python main.py --do_train \
                --train_list dataset/train.txt \
                --test_list dataset/val.txt \
                --model ResNetSE34V2 \
                --max_epoch 500 \
                --batch_size 128 \
                --nDataLoaderThread 2 \
                --criterion amsoftmax \
                --margin 0.1\
                --scale 30\
                --nPerSpeaker 1 \
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
                --criterion angleproto \
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
                --criterion softmaxproto \
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
                --initial_model_infer exp/ResNetSE34V2/model/best_state.model
```

2. Evaluate and tune thresholds

```python
!python main.py --do_infer --eval \
                --model ResNetSE34V2 \
                --test_list dataset/val.txt \
                --cohorts_path checkpoints/cohorts_resnet34v2.npy \
                --initial_model_infer exp/ResNetSE34V2/model/best_state.model
```

3. Run on Test set

```python
!python main.py --do_infer --test \
                --model ResNetSE34V2 \
                --cohorts_path checkpoints/cohorts_resnet34v2.npy \
                --test_threshold 1.7206447124481201 \
                --test_path dataset \
                --initial_model_infer exp/ResNetSE34V2/model/best_state.model
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

### Important arguments:
1. max_frames (default = 100)
2. eval_frames
3. batch_size
4. nDataLoaderThread
5. augment
6. device
7. model
8. max_epoch
9. criterion
10. save_model_last
11. preprocess (default True, set to False if training with E2E model ex: RawNetv2)
12. early_stop
13. es_patience (default 20, if early_stop is set)
14. margin (default 0.2 for amsoftmax, aamsoftmax loss)
15. scale (default 30 for amsoftmax, aamsoftmax loss)
16. scale_pos, scale_neg (for msloss)
17. nPerSpeaker (if classification loss set it to 1 if metric loss set to 2, if msloss its likely to set to 5)
18. nClasses (default 400, change due to the dataset)
19. initial_model
20. train_list (metadata of training files)
21. test_list (metadata of testing files)
22. n_mels
23. encoder_type
24. scoring_mode (using norm or cosine to scoring)
25. initial_model_infer


## Citation

[1] _In defence of metric learning for speaker recognition_

```
@inproceedings{chung2020in,
    title={In defence of metric learning for speaker recognition},
    author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
    booktitle={Interspeech},
    year={2020}
}
```

[2] _Clova baseline system for the VoxCeleb Speaker Recognition Challenge 2020_

```
@article{heo2020clova,
    title={Clova baseline system for the {VoxCeleb} Speaker Recognition Challenge 2020},
    author={Heo, Hee Soo and Lee, Bong-Jin and Huh, Jaesung and Chung, Joon Son},
    journal={arXiv preprint arXiv:2009.14153},
    year={2020}
}
```

[3] _Analysis of score normalization in multilingual speaker recognition_

```
@inproceedings{inproceedings,
    title = {Analysis of Score Normalization in Multilingual Speaker Recognition},
    author = {Matejka, Pavel and Novotny, Ondrej and Plchot, Oldřich and Burget, Lukas and Diez, Mireia and Černocký, Jan},
    booktitle = {Interspeech},
    year = {2017}
}
```
