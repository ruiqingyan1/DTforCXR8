# encoding: utf-8

"""
3、EEEA-Net-C2 + CheXNet
"""

import os
import darmo
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from read_data import ChestXrayDataSet


def main():
    # 1、设备配置，模型参数
    N_CLASSES = 14
    BATCH_SIZE = 16
    device = torch.device("cpu")
    STUDENT_CKPT_PATH = '../1-EEEA-Net-C2/model1.pth'
    TEACHER_CKPT_PATH = '../3-CheXNet/model3.pth'
    DATA_DIR = '../ChestX-ray14/images-224'
    DT_IMAGE_LIST = '../ChestX-ray14/labels/val_list.txt'

    # 2、加载教师模型和学生模型
    student_model = darmo.create_model("eeea_c2", num_classes=1000, pretrained=True)
    student_model.reset_classifier(num_classes=14, dropout=0.2)

    teacher_model = DenseNet121(N_CLASSES)

    # 3、加载模型参数信息
    if os.path.isfile(TEACHER_CKPT_PATH):
        print("=> loading teacher_checkpoint")
        teacher_checkpoint = torch.load(TEACHER_CKPT_PATH, map_location=torch.device('cpu'))
        teacher_model.load_state_dict(teacher_checkpoint)
        print("=> loaded teacher_checkpoint")
    else:
        print("=> no teacher_checkpoint found")

    if os.path.isfile(STUDENT_CKPT_PATH):
        print("=> loading student_checkpoint")
        student_checkpoint = torch.load(TEACHER_CKPT_PATH, map_location=torch.device('cpu'))
        teacher_model.load_state_dict(student_checkpoint)
        print("=> loaded student_checkpoint")
    else:
        print("=> no student_checkpoint found")

    # 4、数据预处理
    dt_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # 先统一调整到256x256
        transforms.RandomResizedCrop(224),  # 再随机裁剪到224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 将图片转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    dt_dataset = ChestXrayDataSet(
        data_dir=DATA_DIR,
        image_list_file=DT_IMAGE_LIST,
        transform=dt_transforms
    )
    dataloader = DataLoader(
        dataset=dt_dataset,
        batch_size=BATCH_SIZE,  # 设置批次大小为512
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 5、优化器
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    # 6、蒸馏参数
    T = 2.0
    alpha = 0.5
    gamma = 2.0  # Focal loss中的gamma参数

    # 7、我们已经定义了学生模型(student_model)和教师模型(teacher_model)
    teacher_model.eval()  # 教师模型设置为评估模式
    student_model.train()  # 学生模型设置为训练模式

    # 8、开始更新相关参数，并保存模型
    num_epochs = 10  # 定义训练的轮数
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, target in dataloader:  # 循环数据加载器
            data, target = data.to(device), target.to(device)
            target = target.float().unsqueeze(1)  # 确保目标是正确的形状

            # 教师模型前向传播
            with torch.no_grad():
                teacher_logits = teacher_model(data)

            # 学生模型前向传播
            student_logits = student_model(data)

            # 计算蒸馏损失
            loss = distillation_loss(student_logits, teacher_logits, target, T, alpha, gamma)

            # 反向传播和优化
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数

            running_loss += loss.item() * data.size(0)

        # 计算并打印平均损失
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # 在循环结束后，你可以保存学生模型的状态字典
    torch.save(student_model.state_dict(), 'modeld3.pth')


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


# 定义知识蒸馏损失函数
def bce_loss(p, y):
    return F.binary_cross_entropy(p, y)


def focal_bce_loss(p_s, y, gamma):
    return -(1 - p_s) ** gamma * y * torch.log(p_s) - p_s ** gamma * (1 - y) * torch.log(1 - p_s)


def mse_loss(p, q, T):
    return T ** 2 * F.mse_loss(F.softmax(p / T, dim=1), F.softmax(q / T, dim=1))


def distillation_loss(p_s, p_t, y, T, alpha, gamma):
    # 计算Focal Binary Cross Entropy Loss
    focal_bce = focal_bce_loss(p_s, y, gamma)

    # 计算MSE Loss
    mse = mse_loss(p_s, p_t, T)

    # 组合两个损失，按照蒸馏损失方程
    loss_kd = alpha * focal_bce + (1 - alpha) * mse
    return loss_kd


if __name__ == '__main__':
    main()