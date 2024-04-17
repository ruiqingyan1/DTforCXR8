# encoding: utf-8

"""
测试 CheXNet 训练好model的效果
"""

import re
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score,accuracy_score, f1_score

CKPT_PATH = 'model3.pth'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = '../ChestX-ray14/images-224'
TRAIN_IMAGE_LIST = '../ChestX-ray14/labels/train_list.txt'
TEST_IMAGE_LIST = '../ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 8


def main():
    # 1、为每个卷积层搜索最适合他的卷计算法，这里有解释https://zhuanlan.zhihu.com/p/73711222
    cudnn.benchmark = True

    # 2、加载模型
    model = DenseNet121(N_CLASSES)

    # 3、加载模型参数信息
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        # checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
        # model.load_state_dict(checkpoint)
        # 如果你使用的是原model3.pth，请使用下面匹配规则
        checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
        new_state_dict = update_state_dict(model,checkpoint)
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    # 4、加载模型参数信息
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        # 这里没有用cuda，同时用了update_state_dict改了一些名字适配，你可能需要调整或者删除这个函数
        checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
        new_state_dict = update_state_dict(model,checkpoint)
        # 将新的state_dict加载到模型中
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    # 4、数据预处理，定义应用于多个裁剪的转换的函数
    def apply_ten_crop(crops):
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])
    def apply_normalize(crops):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return torch.stack([normalize(crop) for crop in crops])
    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda(apply_ten_crop),
                                        transforms.Lambda(apply_normalize)
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    # 5、初始化一个空的FloatTensor对象，用于存储模型的预测结果中的真实标签（ground truth）
    gt = torch.FloatTensor()
    # 6、初始化一个空的FloatTensor对象，用于存储模型的预测结果。
    pred = torch.FloatTensor()

    # 7、模型切换到评估模式
    model.eval()

    # 8、分批次进行预测和验证
    for i, (inp, target) in enumerate(test_loader):
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)
        # 打印当前批次和处理进度
        print(f"Processing batch {i + 1}/{len(test_loader)}, Batch size: {bs}")

    # 9、计算AUC
    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))

    # 10、计算Accuracy和F1
    accuracies, f1_scores = compute_accuracy_f1(gt, pred)
    accuracy_avg = np.mean(accuracies)
    f1_avg = np.mean(f1_scores)
    print('The average Accuracy is {:.3f}'.format(accuracy_avg))
    print('The average F1 Score is {:.3f}'.format(f1_avg))
    for i in range(N_CLASSES):
        print('The Accuracy of {} is {:.3f}'.format(CLASS_NAMES[i], accuracies[i]))
        print('The F1 Score of {} is {:.3f}'.format(CLASS_NAMES[i], f1_scores[i]))


# 如果你使用的是原pth，需要更新匹配规则
def update_state_dict(model,checkpoint):
    state_dict = checkpoint['state_dict']
    # 创建新的state_dict
    new_state_dict = {}

    # 匹配点和数字的组合，以及分类器键名错误
    pattern1 = re.compile(r'(\.\d+)')
    classifier_pattern = re.compile(r'classifier(\d)')
    pattern2 = re.compile(r'module\.(.*)')
    for key, value in state_dict.items():
        # 替换格式错误的键名（例如 'norm.1' 替换为 'norm1'）
        new_key = re.sub(pattern1, lambda x: x.group().replace('.', ''), key)
        # 针对分类器部分的特定替换
        new_key = re.sub(classifier_pattern, r'classifier.\1', new_key)
        # 删除module.的前缀
        new_key = re.search(pattern2,new_key)
        # new_state_dict[new_key] = value
        if new_key:
            new_key = new_key.group(1)
            new_state_dict[new_key] = value
    return new_state_dict


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def compute_accuracy_f1(gt, pred, average='macro'):
    """
    Computes Accuracy and F1 score from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
            true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
            can either be probability estimates of the positive class,
            confidence values, or binary decisions.
        average: String, [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
            This parameter is required for multiclass/multilabel targets.
            If `None`, the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the data.

    Returns:
        Tuple containing lists of accuracies and F1 scores for each class.
    """
    accuracies = []
    f1_scores = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()

    # Assuming pred_np contains probabilities, threshold to get binary predictions
    pred_np = (pred_np > 0.5).astype(int)

    for i in range(N_CLASSES):
        accuracies.append(accuracy_score(gt_np[:, i], pred_np[:, i]))
        f1_scores.append(f1_score(gt_np[:, i], pred_np[:, i], average=average))

    return accuracies, f1_scores


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    main()