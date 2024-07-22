# Domain-Collaborative Contrastive Learning for Hyperspectral Image Classification
This repository provides code for "Domain-Collaborative Contrastive Learning for Hyperspectral Image Classification" GRSL-2024. ([Paper](https://ieeexplore.ieee.org/abstract/document/10589700/))

## One-sentence description
In this paper, we propose DCCL for cross-domain HSIC. DCCL collaboratively utilizes the labeled examples in source domain and the class centers in target domain to generate high-confidence pseudo labels. Finally, contrastive learning is used to reduce domain gap.

## Requirements
Note that DCCL is only tested on Ubuntu OS 18.04.5 with the following environments. It may work on other operating systems as well but we do not guarantee that it will.<br>
#### CUDA 10.1<br>
#### cuDNN 7.6.5<br>
#### Python 3.6.13<br>
#### Pytorch 1.6.0<br>
#### scikit-learn 0.24.2<br>
#### numpy 1.19.2

## Citation
Please cite our paper if you find the work useful:
```
@article{luo2024domain,
  title={Domain-Collaborative Contrastive Learning for Hyperspectral Image Classification},
  author={Luo, Haiyang and Qiao, Xueyi and Xu, Yongming and Zhong, Shengwei and Gong, Chen},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgements
Some codes are adapted from [BCDM](https://github.com/BIT-DA/BCDM/tree/master) and [CLDA](https://github.com/Li-ZK/CLDA-2022/tree/main). We thank them for their excellent projects.
