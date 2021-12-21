# DCMF
# Learning Discriminative Cross-modality Features for RGB-D Saliency Detection (DCMF)
source code for our TIP 2021 paper "Learning Discriminative Cross-modality Features for RGB-D Saliency Detection" by Fengyun Wang, Jinshan Pan, Shoukun Xu, and Jinhui Tang

created by Fengyun Wang, email: fereenwong@gmail.com

![avatar](https://github.com/fereenwong/DCMF/blob/main/Overview.png)

## Requirement
1. Pytorch 1.7.0 (a lower vision may also workable)
2. Torchvision 0.7.0

## Data Preparation
**Training**: with 1400 images from NJU2K, 650 images from NLPR, and 800 images from DUT-RGBD (And 100 images from NJU2K and 50 images from NLPR for validation).

**Testing**: with 485 images from NJU2K, 300 images from NLPR, 400 images from DUT-RGBD, 1000 images from STERE, 1000 images from ReDWeb-S, 100 images from LFSD, and 80 images from SSD.

You can directly download these dataset from here: [NJU2K]()   [NLPR]()   [DUT-RGBD]()   [STERE]()   [ReDWeb-S]()   [LFSD]()   [SSD]()

After downloading, put them into `your_RGBD_Dataset` folder, and it should look like this:
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
1.  Download the pretrained VGG model [[baidu pan](https://pan.baidu.com/s/1ec39Ld3trwk2j1dQsfKzBA) fetch code: 44be | [google drive](https://drive.google.com/file/d/1LRfwb-LbJaPmvCAWr6_3YubM1py3GOD_/view?usp=sharing)] and put it into `./pretrained_model` folder.
2.  Run `python train.py your_RGBD_Dataset` for training.

## Testing on Our Pretrained model
1. Download our pretrained model [[baidu_pan](https://pan.baidu.com/s/1VbkWoMTMTvSDTsu3h1KWhA) fetch_code:kc76 | [google_drive](https://drive.google.com/file/d/1bjQLtDsmrYczGj-nb9hh9M3GpsVM6zTr/view?usp=sharing)] and then put it in `./checkpoint` folder.
2. Run `python test.py ./checkpoint/corr_pac.pth your_RGBD_Dataset`. The predictions will be in `./output` folder.

## Ours Saliency Maps
- [NJU2K]()   
- [NLPR]()   
- [DUT-RGBD]()   
- [STERE]()   
- [ReDWeb-S]()   
- [LFSD]()   
- [SSD]()

## Citation
If you think our work is helpful, please cite 
```

```
