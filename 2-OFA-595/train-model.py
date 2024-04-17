import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
import darmo
import time


CKPT_PATH = 'model2.pth'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = '../ChestX-ray14/images-224'
TRAIN_IMAGE_LIST = '../ChestX-ray14/labels/train_list_224.txt'
TEST_IMAGE_LIST = '../ChestX-ray14/labels/test_list_244.txt'
BATCH_SIZE = 16


def main():
    # 1、为每个卷积层搜索最适合他的卷计算法，这里有解释https://zhuanlan.zhihu.com/p/73711222
    cudnn.benchmark = True

    # 2、加载模型
    model = darmo.create_model("ofa595_1k", num_classes=1000, pretrained=True)
    model.reset_classifier(num_classes=14, dropout=0.2)

    # 3、数据预处理
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # 先统一调整到256x256
        transforms.RandomResizedCrop(224),  # 再随机裁剪到224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 将图片转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    train_dataset = ChestXrayDataSet(
        data_dir=DATA_DIR,
        image_list_file=TRAIN_IMAGE_LIST,
        transform=train_transforms
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,  # 设置批次大小为512
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 4、定义了损失函数 BCEWithLogitsLoss 和优化器 AdamW
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 5、训练model，保存参数
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)


def train_model(model, data_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_loss = float('inf')
    total_samples = len(data_loader.dataset)
    batches = len(data_loader)

    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()  # 开始时间

        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            # 每处理完一个batch输出一次进度和该batch的损失
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{batches}, Batch Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total_samples
        end_time = time.time()  # 结束时间
        epoch_duration = end_time - start_time  # 计算一个epoch的持续时间
        print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {epoch_loss:.4f}, Time: {epoch_duration:.2f} sec")

        # 如果当前 epoch 的损失小于之前的最佳损失，则保存模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            base_filename = 'model2'
            filename = f'{base_filename}.pth'
            counter = 2

            # 检查文件是否存在，并更新文件名
            while os.path.exists(filename):
                filename = f'{base_filename}_{counter}.pth'
                counter += 1

            # 保存模型参数到文件
            torch.save(model.state_dict(), filename)
            print(f"Model saved as {filename}")


if __name__ == '__main__':
    main()