
# DTforCXR8

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eeea-net-an-early-exit-evolutionary-neural/neural-architecture-search-on-cifar-10)](https://paperswithcode.com/sota/neural-architecture-search-on-cifar-10?p=eeea-net-an-early-exit-evolutionary-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eeea-net-an-early-exit-evolutionary-neural/object-detection-on-pascal-voc-2007)](https://paperswithcode.com/sota/object-detection-on-pascal-voc-2007?p=eeea-net-an-early-exit-evolutionary-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eeea-net-an-early-exit-evolutionary-neural/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=eeea-net-an-early-exit-evolutionary-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eeea-net-an-early-exit-evolutionary-neural/neural-architecture-search-on-imagenet)](https://paperswithcode.com/sota/neural-architecture-search-on-imagenet?p=eeea-net-an-early-exit-evolutionary-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eeea-net-an-early-exit-evolutionary-neural/image-classification-on-cifar-100)](https://paperswithcode.com/sota/image-classification-on-cifar-100?p=eeea-net-an-early-exit-evolutionary-neural)

《基于知识蒸馏的轻量化迁移模型在肺部疾病诊断中的应用研究》

《Application of Lightweight Migration Model Based on Knowledge Distillation in the Diagnosis of Lung Diseases》

## 0、简介(Introduction)

🌺在EEEA-Net-C2、OFA-595、Visformer-small和AutoFormerV2-base使用ImageNet_1k的相关内容直接进行10epoch的训练，拿到性能比较好的model.pth

🌻在CheXNet使用预训练好的效果比较好的mode.pth，进行10epoch的训练，拿到性能比较好的model.pth后保存

🏵依次拿到上面的模型训练参数model1.pth、model2.pth、model3.pth、model4.pth、model5.pth，进行知识蒸馏，拿到对应的EEEA-Net-C2蒸馏训练参数modeld1.pth、modeld2.pth、modeld3.pth、modeld4.pth、modeld5.pth

🌹分析原模型和蒸馏后模型的AUC、Accuracy、F1进行对比得出实验结果，下面是实验结果

## 1、使用（Usage）

克隆项目到本地（Clone the repository）

```
git clone https://github.com/ruiqingyan1/DTforCXR8.git
```

配置实验环境（Configuration of experimental environment）

```bash
conda create -n openmmlab python=3.8
conda activate openmmlab
pip install -r requirements.txt
```

数据集下载（Download CXR8）

NIH提供了数据集下载：[NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)，kaggle提供了数据集下载[NIH](https://www.kaggle.com/datasets/nih-chest-xrays/data/discussion/300917)，可以把12个images文件合并方便操作。

不想合并[kaggle也提供了数据预处理](https://www.kaggle.com/code/sbernadac/lung-deseases-data-analysis)的形式，需要重写train_list.txt和test_list.txt，把文件名该成相对路径。

kaggle提供了[CXR-resize-224的数据集](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)，直接使用训练会更快。

## 2、原模型训练、测试（Basic train&test）

以1-EEEA-Net-C2为例，2-OFA-595、3-CheXNet、4-Visformer-small、5-AutoformerV2-base运行方法相同，请先通过train-model.py训练出自己的model.pth，之后通过test-model.py拿到自己的AUC等测试信息。

### 2.1 EEEA-Net-C2相关信息

在1-EEEA-Net-C2文件夹内，运行para-test-eeeac2.py，可以看到模型的参数信息

```test
统计模型 parameters 参数量，The total number of parameters: 4732838，4.732838M
统计模型 FLOPs 量，Total FLOPs: 311335744.0，3.11亿
```

### 2.2 在ImageNet_1k参数基础上，进行模型训练

在1-EEEA-Net-C2文件夹内，运行train-model.py，可以看到开始训练模型，你可以调小BATCH_SIZE到16，电脑性能问题BATCH_SIZE在512的话需要长时间才能看到相关信息

### 2.3 拿到训练好的model1.pth，测试训练结果

在1-EEEA-Net-C2文件夹内，运行test-model.py，可以看到开始训练模型，你可以调小BATCH_SIZE到16，电脑性能问题BATCH_SIZE在512的话需要长时间才能看到相关信息

### 2.4 实验结果，以AUC为例

```test
# The average AUROC is 0.823
# The AUROC of Atelectasis is 0.807
# The AUROC of Cardiomegaly is 0.881
# The AUROC of Effusion is 0.789
# ...
```



## 3、蒸馏模型训练、测试（DT train&test）

在6-Distilling-to-EEEA-Net-C2文件夹内



