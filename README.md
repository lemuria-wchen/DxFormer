## DxFormer

Code for our Bioinformatics 2022 paper: [DxFormer: A Decoupled Automatic Diagnostic System Based on Decoder-Encoder Transformer with Dense Symptom Representations](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac744/6835407). 
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
## Citation

If you use or extend this work, please cite [this paper](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac744/6835407) where it is introcuded. 

```
@article{10.1093/bioinformatics/btac744,
    author = {Chen, Wei and Zhong, Cheng and Peng, Jiajie and Wei, Zhongyu},
    title = "{DxFormer: a decoupled automatic diagnostic system based on decoder–encoder transformer with dense symptom representations}",
    journal = {Bioinformatics},
    year = {2022},
    month = {11},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac744},
    url = {https://doi.org/10.1093/bioinformatics/btac744},
    note = {btac744},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btac744/47804760/btac744.pdf},
}
```

