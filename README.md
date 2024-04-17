
# DTforCXR8

《基于知识蒸馏的轻量化迁移模型在肺部疾病诊断中的应用研究》

《Application of Lightweight Migration Model Based on Knowledge Distillation in the Diagnosis of Lung Diseases》

## 0、简介(Introduction)

🌺在EEEA-Net-C2、OFA-595、Visformer-small和AutoFormerV2-base使用ImageNet_1k的相关内容直接进行10epoch的训练，拿到性能比较好的model.pth

🌻在CheXNet使用预训练好的效果比较好的[mode.pth](https://github.com/arnoweng/CheXNet)，进行10epoch的训练，拿到性能比较好的model3.pth后保存

🏵依次拿到上面的模型训练参数model1.pth、model2.pth、model3.pth、model4.pth、model5.pth，进行知识蒸馏，拿到对应的EEEA-Net-C2蒸馏训练参数modeld1.pth、modeld2.pth、modeld3.pth、modeld4.pth、modeld5.pth

🌹分析原模型和蒸馏后模型的AUC、Accuracy、F1进行对比得出实验结果，下面是实验结果

## 1、使用（Usage）

🖼克隆项目到本地（Clone the repository）

```
git clone https://github.com/ruiqingyan1/DTforCXR8.git
```

🖼配置实验环境（Configuration of experimental environment）

```bash
conda create -n openmmlab python=3.8
conda activate openmmlab
pip install -r requirements.txt
```

🖼数据集下载（Download CXR8）

> ⭕数据集中图片文件到./CheXNet-RAY14/images相关目录下，可以参考代码内的
>
> DATA_DIR = '../ChestX-ray14/images'，数据集下载链接如下

NIH提供了数据集下载：[NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)，kaggle提供了数据集下载[NIH](https://www.kaggle.com/datasets/nih-chest-xrays/data/discussion/300917)，可以把12个images文件合并方便操作。

不想合并[kaggle也提供了数据预处理](https://www.kaggle.com/code/sbernadac/lung-deseases-data-analysis)的形式，需要重写train_list.txt和test_list.txt，把文件名该成相对路径。

kaggle提供了[CXR-resize-224的数据集](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)，直接使用训练会更快。

## 2、原模型训练、测试（Basic train&test）

以1-EEEA-Net-C2为例，2-OFA-595、3-CheXNet、4-Visformer-small、5-AutoformerV2-base运行方法相同，请先通过train-model.py训练出自己的model.pth，之后通过test-model.py拿到自己的AUC等测试信息。

### 2.1 EEEA-Net-C2相关信息

在1-EEEA-Net-C2文件夹内，运行para-test-eeeac2.py，可以看到模型的参数信息

> 统计模型 parameters 参数量，The total number of parameters: 4732838，4.732838M
> 统计模型 FLOPs 量，Total FLOPs: 311335744.0，3.11亿

### 2.2 在ImageNet_1k参数基础上，进行模型训练

在1-EEEA-Net-C2文件夹内，运行train-model.py，可以看到开始训练模型，你可以调小BATCH_SIZE到16，电脑性能问题BATCH_SIZE在512的话需要长时间才能看到相关信息，训练好后拿到model1.pth。

```bash
cd 1-EEEA-Net-C2
python train-model.py
```

### 2.3 拿到训练好的model1.pth，测试训练结果

在1-EEEA-Net-C2文件夹内，运行test-model.py，可以看到开始测试模型，你可以调小BATCH_SIZE到16，电脑性能问题BATCH_SIZE在512的话需要长时间才能看到相关信息，之后拿到测试结果。

```bash
cd 1-EEEA-Net-C2
python test-model.py
```

### 2.4 实验结果，以AUC为例

> The average AUROC is 0.802
>
> The AUROC of Atelectasis is 0.825
>
> The AUROC of Cardiomegaly is 0.913
>
> The AUROC of Effusion is 0.842
>
> The AUROC of Infiltration is 0.722
>
> ...

## 3、蒸馏模型训练、测试（DT train&test）

以train-kd3-model.py为例，其中kd1~5分别代表五个不同模型蒸馏到EEEA-Net-C2，分别为train-kd1-model.py、train-kd2-model.py、train-kd3-model.py、train-kd4-model.py、train-kd5-model.py，训练后可以得到modeld1.pth、modeld2.pth、modeld3.pth、modeld4.pth、modeld5.pth，在想对应的测试模块可以检验训练效果

### 3.1 CheXNet蒸馏到EEEA-Net-C2

在6-Distilling-to-EEEA-Net-C2文件夹内，运行train-kd3-model.py，开始从EEEA-Net-C2蒸馏到EEEA-Net-C2，蒸馏后的EEEA-Net-C2参数信息保存在modeld1.pth

```bash
cd 6-Distilling-to-EEEA-Net-C2
python train-kd3-model.py
```

### 3.2 测试蒸馏效果

在6-Distilling-to-EEEA-Net-C2文件夹内，运行test-kd3-model.py，测试modeld3.pth参数下的EEEA-Net-C2在CXR分类上的性能

```
python train-kd3-model.py
```

### 3.3 实验结果，以AUC为例

> The average AUROC is 0.837
>
> The AUROC of Atelectasis is 0.824
>
> The AUROC of Cardiomegaly is 0.916
>
> The AUROC of Effusion is 0.886
>
> The AUROC of Infiltration is 0.710
>
> ...

## 4、结果分析

<img src="/Users/ruiqingyan/Library/Application Support/typora-user-images/image-20240417183241716.png" alt="image-20240417183241716" style="zoom:50%;" />

ChestX-ray14数据集上不同模型的Mean AUC对比来看，在我们的实验中，通过对未蒸馏的基模型和蒸馏后的模型在ChestX-ray14数据集上的性能进行详细比较，我们发现CheXNet作为教师模型，EEEA-Net-C2作为学生模型的知识蒸馏策略表现最为出色。具体而言，未经蒸馏的CheXNet在所有基模型中展现了最高的AUC值（83.8%），表明其具有卓越的疾病分类能力。而在知识蒸馏后，采用CheXNet作为教师模型，EEEA-Net-C2作为学生模型进一步提升了性能，AUC值达到了83.8%，表现出了优化后的模型确实能够接近甚至超越未蒸馏模型的性能。这一结果不仅验证了知识蒸馏技术在提高轻量化模型性能中的有效性，同时也证明了CheXNet和EEEA-Net-C2组合的蒸馏策略在优化模型性能方面具有显著的优势。通过这种方法，我们成功地构建了一个既高效又紧凑的模型，为资源有限的医疗环境提供了一个强大的诊断工具，推动了医疗图像分析技术的应用和发展。此外，其他模型如OFA-595、Visformer-small和AutoFormerV2-base在蒸馏后也显示出了性能的提升，这进一步强调了蒸馏技术对于多种不同架构的模型都有益。
