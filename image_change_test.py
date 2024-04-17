from torchvision import transforms
from PIL import Image

# 载入图像文件
image_path = '/Volumes/ruiqingyan/input/images_001/00000001_000.png'  # 请将此路径替换为您的图像文件路径
image = Image.open(image_path)

# 定义转换操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor()           # 将图像转换为Tensor
])

# 应用转换操作
transformed_image = transform(image)

# 将Tensor转换回PIL图像，以便保存
to_pil = transforms.ToPILImage()
saved_image = to_pil(transformed_image)

# 保存图像
save_path = 'transformed_imag.jpg'  # 保存文件的路径
saved_image.save(save_path)
print(f"Image saved to {save_path}")
