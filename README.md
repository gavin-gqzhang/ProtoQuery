# Query-Guided Predicate Decoupling and Prototype Approximation Learning for Scene Graph Generation

This repository contains the official code implementation for the paper "Query-Guided Predicate Decoupling and Prototype Approximation Learning for Scene Graph Generation"

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Train
We provide [scripts](./scripts/train.sh) for training the models

### <font color="red">If you have any questions, please contact me: guoqing.zhang@bjtu.edu.cn.</font>

## Device

All our experiments are conducted on four NVIDIA GeForce RTX 3090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).


## Logs

<!-- Due to random seeds and machines, they are not completely consistent with those reported in the paper, but they are within the allowable error range. -->
### <font color="red">We will upload the trained models and log files gradually.</font>


|      Model       | Dataset | R@50  | R@100 | mR@50 | mR@100 | F@50 | F@100 |                         Log Path                         |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :----: | :----------------------------------------------------------: |
| PE-Net-ProtoQuery (PredCls) | VG | 57.08 | 60.41 | 38.92 | 42.69 | 46.28 | 50.03  | [Log Link](./logs/PENet-DPPLML-VG-predcls.log) |
| Transformer-ProtoQuery (PredCls) | VG | 61.92 | 63.92 | 36.20 | 39.00 | 45.69 | 48.45  | [Log Link](./logs/Transformer-DPPLML-VG-predcls.log) |
| PE-Net-ProtoQuery (PredCls) | GQA | 48.4 | 51.42 | 34.59 | 36.53 | 40.35 | 42.71  | [Log Link](./logs/PENet-DPPLML-GQA-predcls.log) |


## Tips

We use the `rel_nms` [operation](./maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) provided by [RU-Net](https://github.com/siml3/RU-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) and [HL-Net](https://github.com/siml3/HL-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) in PredCls and SGCls to filter the predicted relation predicates, which encourages diverse prediction results. 


## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [PE-Net](https://github.com/VL-Group/PENET).

