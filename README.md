# Welcome use Wheat Seedling Former

Welcome to Wheat Seedling Former, we will tell you how to use each module of the application, as well as the environment configuration of the software.

## Installation
To run.py files, you need to install the following libraries
```javascript
conda create -n wheatformer python=3.8
conda activate wheatformer
pip install -r requirements.txt
pip install -e .
```
## How to train a model
If you wish to train a model, the following steps can be taken.
```javascript
cd /File path/Wheat-Seedling-Former-
python tools/train.py {File Path}\Wheat-Seedling-Former-\configs\segformer\segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py
```

## Peroration
We mainly added SCSA.py, asp.py, and cfpt.py to mmseg\models\decode_heads, and modified segformer_head.py.The above represents merely the first version of the software, and suggestions will be gathered and sorted out in the future for the improvement of the software.


# AI-PheneLab
