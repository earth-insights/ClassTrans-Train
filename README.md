# Class Similarity Transition: Decoupling Class Similarities and Imbalance from Generalized Few-shot Segmentation

This project is used to train the stage1 of GFSS. We suggest to skip this step and directly use this **[checkpoint](https://drive.google.com/file/d/1H9Z9bLU46tDoqXHEhc4BduQ_Vs2RqGvM/view?usp=sharing)** to reimplement our results.


> **Abstract:** *In Generalized Few-shot Segmentation (GFSS), a model is trained with a large corpus of base class samples and then adapted on limited samples of novel classes. This paper focuses on the relevance between base and novel classes, and improves GFSS in two aspects: 1) mining the similarity between base and novel classes to promote the learning of novel classes, and 2) mitigating the class imbalance issue caused by the volume difference between the support set and the training set. Specifically, we first propose a similarity transition matrix to guide the learning of novel classes with base class knowledge. Then, we leverage the Label-Distribution-Aware Margin (LDAM) loss and Transductive Inference to the GFSS task to address the problem of class imbalance as well as overfitting the support set. In addition, by extending the probability transition matrix, the proposed method can mitigate the catastrophic forgetting of base classes when learning novel classes. With a simple training phase, our proposed method can be applied to any segmentation network trained on base classes. We validated our methods on the adapted version of OpenEarthMap. Compared to existing GFSS baselines, our method excels them all from 3\% to 7\% and ranks second in the OpenEarthMap Land Cover Mapping Few-Shot Challenge at the completion of this paper.*

## &#x1F3AC; Getting Started

### :one: Download data

#### Pre-processed data from drive

We provide the versions of OpenEarthMap datasets used in this work [here](https://zenodo.org/records/10828417). You can download the full .zip and directly extract it in the `data/` folder.

#### From scratch

Alternatively, you can prepare the datasets yourself. Here is the structure of the data folder for you to reproduce:

```
data
├── trainset
│   ├── images
│   └── labels
|
├── testset
|   ├── images
|   └── labels
|
└── train.txt


```

### :two: Download pre-trained models

#### Pre-trained backbone and models
We provide the pre-trained backbone and models at - https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/tree/main. You can download them and directly extract them at the root of `pretrain/`.

## &#x1F5FA; Overview of the repo

Data are located in `data/` contains the train dataset. All the codes are provided in `src/`. Testing script is located at the root of the repo.

## &#x2699; Training 

```bash
python train.py 
```


### &#x1F4CA; Output

The weights of the model are saved in the weight/ directory, and you can use them in ClassTrans.



















