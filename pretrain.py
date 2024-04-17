import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data_save import ChestXrayDataSet



N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
# NIH下载的数据集，预处理后放到image-224文件夹，之后进行其他处理
DATA_DIR = './ChestX-ray14/images'
SAVE_DIR = './ChestX-ray14/images-224'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
BATCH_SIZE = 512

# 数据预处理
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ChestXrayDataSet(
    data_dir=DATA_DIR,
    save_dir=SAVE_DIR,
    image_list_file=TRAIN_IMAGE_LIST,
    transform=train_transforms
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


