# ThrombiClassification

## So1
First solution is a smple solution based on ResNet152.
It uses:
- resnet152,
- 16 darkest 384×384 patches,
- model based on CoAt lite medium and attention pooling
- 5-fold StratifiedGroupKFold


## So2 

Second solution is based on 1st place Kaggle competition solution: https://www.kaggle.com/code/khyeh0719/mayo-submission/notebook
It uses: 
- swin_large_patch4_window12_384 transformer trained on ImageNet-22K set,
- 16 darkest 384×384 patches,
- model based on CoAt lite medium and attention pooling
- 5-fold StratifiedGroupKFold
