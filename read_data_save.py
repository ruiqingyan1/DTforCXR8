import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None, save_dir=None):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.image_names = []
        self.labels = []

        # 创建保存目录，如果不存在的话
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = [int(i) for i in items[1:]]
                self.image_names.append(os.path.join(data_dir, image_name))
                self.labels.append(label)

        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        image = Image.open(image_name).convert('RGB')

        # 对图像进行预处理
        if self.transform:
            transformed_image = self.transform(image)
            image = transformed_image

        # 保存转换后的图像
        if self.save_dir:
            save_path = os.path.join(self.save_dir, os.path.basename(image_name))
            # 确保转换后的图像是PIL图像格式以便保存
            transformed_image_pil = Image.fromarray(transformed_image.mul(255).byte().cpu().numpy().transpose(1, 2, 0))
            transformed_image_pil.save(save_path)

        # 将图像转换为张量
        if self.transform:
            image = transforms.ToTensor()(image)

        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)



