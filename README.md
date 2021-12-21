# DCMF
# Learning Discriminative Cross-modality Features for RGB-D Saliency Detection (DCMF)
source code for our TIP 2021 paper "Learning Discriminative Cross-modality Features for RGB-D Saliency Detection" by Fengyun Wang, Jinshan Pan, Shoukun Xu, and Jinhui Tang

created by Fengyun Wang, email: fereenwong@gmail.com

![avatar](https://github.com/fereenwong/DCMF/blob/main/Overview.png)

## Requirement
1. Pytorch 1.7.0 (a lower vision may also workable)
2. Torchvision 0.7.0

## data preparation
**Training**: with 1400 images from NJU2K, 650 images from NLPR, and 800 images from DUT-RGBD (And 100 images from NJU2K and 50 images from NLPR for validation).

**Testing**: with 485 images from NJU2K, 300 images from NLPR, 400 images from DUT-RGBD, 1000 images from STERE, 1000 images from ReDWeb-S, 100 images from LFSD, and 80 images from SSD.

You can directly download these dataset from here: [NJU2K]()   [NLPR]()   [DUT-RGBD]()   [STERE]()   [ReDWeb-S]()   [LFSD]()   [SSD]()

After downloading, put them into `RGBD_Dataset` folder, and it should look like this:
````
-- RGBD_Dataset
   |-- NJU2K
   |   |-- trainset
   |   |-- | RGB
   |   |-- | depth
   |   |-- | GT
   |   |-- testset
   |   |-- | RGB
   |   |-- | depth
   |   |-- | GT
   |-- STERE
   |   |-- RGB
   |   |-- depth
   |   |-- GT
   ...
````

## Training
1.  Download the pretrained VGG model [baidu pan]() | [google drive]() and put it into `pretrained_model` folder.
2.  Run `python train.py xxx` for training.

## Citation
If you think our work is helpful, please cite 
```

```
