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