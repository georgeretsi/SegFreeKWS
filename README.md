# SegFreeKWS

Description Coming soon! Code for the paper "Keyword Spotting Simplified: A Segmentation-Free Approach using Character Counting and CTC re-scoring" to be presented in ICDAR 2022.

All critical functions for evaluation are available along with an already trained model in ./saved_models folder.

*Updates:*
Training is also supported via train_words.py, e.g.:
```
python train_words.py -lr 1e-3 --dataset iam --max_epochs 40 --model_save_path './saved_models/temp.pt' 
```
