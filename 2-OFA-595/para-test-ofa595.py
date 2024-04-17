# encoding: utf-8

"""
统计 ofa-595 的相关信息，
统计模型 parameters 参数量，The total number of parameters: 7620446，7.620446M
统计模型 FLOPs 量，Total FLOPs: 543255984.0，5.43亿
"""

import darmo
import torch
import torch.backends.cudnn as cudnn
from thop import profile


CKPT_PATH = 'model2.pth'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = '../ChestX-ray14/images-224'
TRAIN_IMAGE_LIST = '../ChestX-ray14/labels/train_list.txt'
TEST_IMAGE_LIST = '../ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 512


def main():
    # 1、为每个卷积层搜索最适合他的卷计算法，这里有解释https://zhuanlan.zhihu.com/p/73711222
    cudnn.benchmark = True

    # 2、加载模型
    model = darmo.create_model("ofa595_1k", num_classes=1000, pretrained=True)
    model.reset_classifier(num_classes=14, dropout=0.2)

    # 3、统计模型parameters参数量，7.620446M
    model_structure(model)

    # 4、统计模型FLOPs量，5.43亿
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print(f"Total FLOPs: {flops}")


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


if __name__ == '__main__':
    main()