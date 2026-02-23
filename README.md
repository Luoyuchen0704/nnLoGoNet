# nnLoGoNet: Retinal Vessel Segmentation via Local-Global Feature Fusion and Skeleton Recall Loss

This is the official code of [nnLoGoNet: Retinal Vessel Segmentation via Local-Global Feature Fusion and Skeleton Recall Loss](https://).

## How to Use
- Download and configure [**nnUNet**](https://github.com/MIC-DKFZ/nnUNet)

- Move **nnUNetTrainer_LoGoNet.py** to **.../nnUNet/nnunetv2/training/nnUNetTrainer/** of the configured nnUNet

- Use nnLoGoNet just like nnUNet:
## Download datasets
### 📊 Dataset Overview

The experiments are conducted on both Optical Coherence Tomography Angiography (OCTA) and Color Fundus Photography (CFP) datasets. The details are summarized in the table below:

| Dataset | Train & Val | Test | Image Size (Pixels) | Characteristics / Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **OCTA-SS** | 44 | 11 | $91 \times 91$ | Small sample size; validates feature extraction in few-shot scenarios. |
| **OCTA-3M** | 160 | 40 | $304 \times 304$ | Clearer capillary details; high-definition microvasculature. |
| **OCTA-6M** | 240 | 60 | $400 \times 400$ | Large Field of View (FOV); complex branching & background noise. |
| **DRIVE** | 20 | 20 | $565 \times 584$ | General benchmark; includes various clinical pathologies. |
| **CHASE-DB1** | 20 | 8 | $999 \times 960$ | Images from children; ensures demographic robustness of the model. |
| **HRF** | 25 | 20 | $3504 \times 2336$ | High-resolution; includes healthy, DR, and glaucoma subjects. |


You can first download [DRIVE](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) for quick check.

All datasets as follows:
1. [OCTA-SS](https://datashare.ed.ac.uk/handle/10283/3528)
2. [OCTA-500(OCTA-3M/6M)](https://ieee-dataport.org/open-access/octa-500)
3. [DRIVE](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction)
4. [CHASE-DB1](https://www.kaggle.com/datasets/khoongweihao/chasedb1)
5. [HRF](https://www5.cs.fau.de/research/data/fundus-images/)


> Data Preprocessing
```
nnUNetv2_plan_and_preprocess -d 11 --verify_dataset_integrity
```

> Training
```
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 11 2d 0 -tr nnUNetTrainer_LoGoNet
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 11 2d 1 -tr nnUNetTrainer_LoGoNet
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 11 2d 2 -tr nnUNetTrainer_LoGoNet
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 11 2d 3 -tr nnUNetTrainer_LoGoNet
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 11 2d 4 -tr nnUNetTrainer_LoGoNet
```

> Continue Training
```
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 11 2d 0 -tr nnUNetTrainer_LoGoNet --c
```

> Best Configuration
```
nnUNetv2_find_best_configuration 11 -c 2d -tr nnUNetTrainer_LoGoNet
```

> Testing
```
nnUNetv2_predict -i .../nnUNetFrame/nnUNet_raw/Dataset11_your_dataset/imagesTs/ -o .../your_predict_path/ -d 11 -c 2d -tr nnUNetTrainer_LoGoNet
```


## Datasets Images



## The Overview of nnLoGoNet.



## Quantitative Comparison



## Qualitative Comparison
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/nnWNet/blob/main/figure/Qualitative%20results%20of%20different%20models%20on%202D%20datasets.png" width="100%" >
<br>(a) Raw images. (b) Ground truth. (c) TransAttUNet. (d) nnUNet. (e) nnWNet.
</p>
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/nnWNet/blob/main/figure/Qualitative%20results%20of%20different%20models%20on%203D%20datasets.png" width="100%" >
<br>(a) Raw images. (b) Ground truth. (c) CoTr. (d) nnUNet. (e) nnWNet.
</p>

## Citation
>If our work is useful for your research, please cite our paper:
```
```
