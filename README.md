# DCMF
# Learning Discriminative Cross-modality Features for RGB-D Saliency Detection (DCMF)
source code for our TIP 2021 paper "Learning Discriminative Cross-modality Features for RGB-D Saliency Detection" by Fengyun Wang, Jinshan Pan, Shoukun Xu, and Jinhui Tang ([PDF]())

created by Fengyun Wang, email: fereenwong@gmail.com

![avatar](https://github.com/fereenwong/DCMF/blob/main/Overview.png)

## Requirement
1. Pytorch 1.7.0 (a lower vision may also workable)
2. Torchvision 0.7.0

## Data Preparation
**Training**: with 1400 images from NJU2K, 650 images from NLPR, and 800 images from DUT-RGBD (And 100 images from NJU2K and 50 images from NLPR for validation).

**Testing**: with 485 images from NJU2K, 300 images from NLPR, 400 images from DUT-RGBD, 1000 images from STERE, 1000 images from ReDWeb-S, 100 images from LFSD, and 80 images from SSD.

You can directly download these dataset (training and testing) from here: 
- NJU2K [[baidu_pan](https://pan.baidu.com/s/1bwN80Az8oX9owxjoeSokJA) fetch_code:bvrg | [google_drive](https://drive.google.com/file/d/1cqVUJsaX4C7RqZbyCpKrMtVNytjt-dd0/view?usp=sharing)]   
- NLPR [[baidu_pan](https://pan.baidu.com/s/1G7NICb4h5zKdaMrbv0Z0Sg) fetch_code:6a2g | [google_drive](https://drive.google.com/file/d/1WetBhpLfmpEHpd5gQJF12cK2BuOF-rJP/view?usp=sharing)]   
- DUT-RGBD [[baidu_pan](https://pan.baidu.com/s/1etYrT0iNQ1C5gTzYc_k2cA) fetch_code:hqbv | [google_drive](https://drive.google.com/file/d/1VVOS9pcwY6_l208G34YSLLOuZJ7PIT2j/view?usp=sharing)]   
- STERE[[baidu_pan](https://pan.baidu.com/s/15xcP-8Jdq3eBMS5uT9-dVA) fetch_code:ffgx | [google_drive](https://drive.google.com/file/d/1gtaPVdP5MWLbmXdXWiLKo_bXOVQYdKIA/view?usp=sharing)]   
- ReDWeb-S [[baidu_pan](https://pan.baidu.com/s/1mf4C-FFiP03Z2dZ9ihZfNg) fetch_code:zupl | [google_drive](https://drive.google.com/file/d/1k_4TtH-mMlNEPx1CgnWpNu_wDTdAtEoj/view?usp=sharing)] （use testset only）
- LFSD [[baidu_pan](https://pan.baidu.com/s/1xkKhsyB-55F8BgWaYNeDaw) fetch_code:0vx1 | [google_drive](https://drive.google.com/file/d/1U7jqnSefkXxIPqt0uSManRBt5cqjjFJA/view?usp=sharing)]   
- SSD100 [[baidu_pan](https://pan.baidu.com/s/1c84cwRmHBLrnr-mt4jivYg) fetch_code:qs2y | [google_drive](https://drive.google.com/file/d/1DGTmRbfvN_iy09lhhEgYIAjjQWDI7Dg7/view?usp=sharing)]

After downloading, put them into `your_RGBD_Dataset` folder, and it should look like this:
````
-- your_RGBD_Dataset
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
- NJU2K [[baidu_pan](https://pan.baidu.com/s/18lvONSMolqf5CVmZPkVFmg) fetch_code:hxt8 | [google_drive](https://drive.google.com/file/d/1Kml5P5Z3IHntpx8K193AbQtR7LuzJg6o/view?usp=sharing)]   
- NLPR [[baidu_pan](https://pan.baidu.com/s/1KylEgB83pVVMbc3WEDofqQ) fetch_code:h1oe | [google_drive](https://drive.google.com/file/d/1V08IZNDBsu-YU2pICyYWurvvgPxsA1fg/view?usp=sharing)]  
- DUT-RGBD [[baidu_pan](https://pan.baidu.com/s/1t7nxCcXjYUWuwuAzzxJ4TA) fetch_code:vni4 | [google_drive](https://drive.google.com/file/d/18mjKSc5U44rUIqlxbp4dGRmxKiOEOWt6/view?usp=sharing)]   
- STERE[[baidu_pan](https://pan.baidu.com/s/1u7gEufzVvhDborgoe76sTQ) fetch_code:8su3 | [google_drive](https://drive.google.com/file/d/1sQ_4gP6c2poDvIO4CEHgHpvhXewmW0Uq/view?usp=sharing)]   
- ReDWeb-S [[baidu_pan](https://pan.baidu.com/s/14fgANdveMSDH2ATmGqiIoA) fetch_code:27hs | [google_drive](https://drive.google.com/file/d/1x03gJrfI8_g_Ypx5ng5TFXzSDiOqfX-W/view?usp=sharing)]
- LFSD [[baidu_pan](https://pan.baidu.com/s/1VQCRCJTem25MKIE_lXCejA) fetch_code:vapc | [google_drive](https://drive.google.com/file/d/1iePLOXxYKfp7YZZ_1BCgQGvIHHMJ-rXS/view?usp=sharing)]  
- SSD100 [[baidu_pan](https://pan.baidu.com/s/1o1aW5tJgxnJg6KRpFZfXNg) fetch_code:2y3i | [google_drive](https://drive.google.com/file/d/1mVYAuSDhl0ErxBabtIRw-sOWAlPDkm5O/view?usp=sharing)]

## Citation
If you think our work is helpful, please cite 
```
COMING SOON ...
```
