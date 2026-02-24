## NYC-SCD dataset
As part of this work, we introduce and release a real-world airborne LiDAR semantic change detection dataset, termed the New York City Semantic Change Dataset (NYC-SCD).

### Dataset Request Procedure
To request access to the NYC-SCD dataset, please complete the following form:
[NYC-SCD Dataset Request Form](https://docs.google.com/forms/d/e/1FAIpQLSeFU5O18UQjvJKDqa1ULxn2xQT-yuvNb1G71EGYyB3US4CaxA/viewform?usp=header)

The download link will be provided via email upon approval.

We provide two versions of the dataset:

1️⃣ version_training
This version is pre-processed and organized for direct use in model training and evaluation.

2️⃣ version_all
This version contains the complete .las files covering the full spatial extent of the 2010 and 2017 acquisition years, including all areas corresponding to the train, validation, and test regions.
## Dataset Overview
The new republic 3D semantic change detection dataset is sourced from high-quality ALS data collected in 2014 and 2017, released under the Open Geospatial Data Program by the New York City government. The total area covered by these scenes is 22.5 km².
## Data description
In the NYC-SCD dataset, ALS point clouds from epochs T0 and T1 are assigned semantic labels, while the T1 data are further annotated with semantic change labels.
The semantic labels of the two epoch point clouds are annotated into four categories, with the following classification for each category:
Ground: road surfaces, pavement, flat terrain, etc.
Building: all man-made structures.
Vegetation: trees and other low-growing plants.
Clutter: ground objects that are not vegetation.
Based on  semantic labels and rules for change detection, the ${T_1}$ epoch point cloud is annotated with four change categories. The categories are as follows:
Unchanged: all semantic categories that have not undergone any changes.
Newly built: newly constructed buildings and added roof attachments.
Demolition: areas where buildings have been demolished.
New clutter: newly appearing objects that include semantic categories such as vegetation and clutter.

## Installation

### Requirements
- Ubuntu: 18.04 and above.
- CUDA: 11.3 and above.
- PyTorch: 1.10.0 and above.

### Conda Environment

```bash
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d
```

## Data Preparation
We provide configuration files, as well as the corresponding training and testing code, for the following three datasets:
Urb3DCD
SLPCCD
NYC-SCD
### Dataset Download Links
Urb3DCD
Urban Point Clouds Simulated Dataset for 3D Change Detection
https://ieee-dataport.org/open-access/urb3dcd-urban-point-clouds-simulated-dataset-3d-change-detection
SLPCCD
https://github.com/wangle53/3DCDNet

## Quick Start

### Training
The training processing is based on configs in `configs` folder. 
For example:
```bash
# Direct
export PYTHONPATH=./
python tools/train.py --config-file configs/cd/Urb3dcd.py --options save_path=./results
```
### Testing
The training processing is based on configs in `configs` folder. 
For example:
```bash
# Direct
export PYTHONPATH=./
python tools/test.py --config-file configs/cd/Urb3dcd.py --options save_path=./results
```

## Acknowledgement
Our ME-CPT work was developed based on the codebase of Point Transformer V3.
- **Point Transformer V3: Simpler, Faster, Stronger**  
*Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao*  
IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) 2024  
[ Backbone ] [PTv3] - [ [arXiv](https://arxiv.org/abs/2312.10035) ] [ [Bib](https://xywu.me/research/ptv3/bib.txt) ] [ [Project](https://github.com/Pointcept/PointTransformerV3) ] &rarr; [here](https://github.com/Pointcept/PointTransformerV3)
