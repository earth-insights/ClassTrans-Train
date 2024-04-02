# Class Similarity Transition: Decoupling Class Similarities and Imbalance from Generalized Few-shot Segmentation

This project is used to train the base class model for [ClassTrans](https://github.com/earth-insights/ClassTrans). We suggest to skip this step and directly use this **[checkpoint](https://drive.google.com/file/d/1H9Z9bLU46tDoqXHEhc4BduQ_Vs2RqGvM/view?usp=sharing)** to reimplement our results.

## &#x1F3AC; Getting Started

### :one: Download data

#### Pre-processed data from drive

We use a [adapted version](https://zenodo.org/records/10828417) of OpenEarthMap datasets. You can download the full .zip and directly extract it in the `data/` folder.

#### From scratch

Alternatively, you can prepare the datasets yourself. Here is the structure of the data folder for you to reproduce:

```
data
├── trainset
│   ├── images
│   └── labels
|
|
└── train.txt
```

### :two: Download pre-trained models

#### Pre-trained backbone and models
We use ConvNext_large pre-trained using CLIP as backbone. You can download the weight [here](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/tree/main) and move it to `pretrain/`.

## &#x1F5FA; Overview of the repo

Data are located in `data/` contains the train dataset. All the codes are provided in `src/`. Testing script is located at the root of the repo.

## &#x2699; Training 

```bash
python train.py 
```

### &#x1F4CA; Output

The weights of the model are saved in the `weight/` and you can use them in [ClassTrans](https://github.com/earth-insights/ClassTrans/tree/main).



















