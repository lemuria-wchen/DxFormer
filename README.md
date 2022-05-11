## DxFormer

Code for paper: [DxFormer: A Decoupled Automatic Diagnostic System Based on Decoder-Encoder Transformer with Dense Symptom Representations](https://arxiv.org/abs/2205.03755). 
### Download data

Download the datasets, then decompress them and put them in the corrsponding documents in  `data/{dxy|mz4|mz10}/raw`. For example, download mz4 dataset and put the dataset to the `data/mz4/raw`. 

The dataset can be downloaded as following links:

- [Dxy dataset](https://github.com/HCPLab-SYSU/Medical_DS)
- [MZ-4 dataset](http://www.sdspeople.fudan.edu.cn/zywei/data/acl2018-mds.zip)
- [MZ-10 dataset](https://github.com/lemuria-wchen/imcs21)

### Preprocess

```shell
python preprocess.py
```

### Accuracy Bound

The default dataset is MZ-10, please modify the code to change dataset by just replace `mz10` to `dxy` or `mz4`. 

```shell
python bound.py
```

### Pre-training

```
python pretrain.py
```

### Training & Inference

```
python train.py
```
