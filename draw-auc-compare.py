import matplotlib.pyplot as plt
import numpy as np

# 模型名称和对应的AUC数据
models = ['Baseline', 'EEEA-Net-C2', 'OFA-595', 'CheXNet', 'Visformer-small', 'AutoFormerV2-base']
auc_pre = [80.2, 80.2, 81.7, 83.7, 82.1, 81.5]
auc_post = [80.2, 81.4, 82.6, 83.8, 82.3, 82.5]

x = np.arange(len(models))  # 模型标签的位置

plt.figure(figsize=(10, 5))  # 设置图形大小
plt.plot(models, auc_pre, marker='o', label='Pre-Distillation')  # 绘制蒸馏前的折线图
plt.plot(models, auc_post, marker='o', label='Post-Distillation')  # 绘制蒸馏后的折线图

plt.ylabel('AUC (%)')  # y轴标签
plt.title('AUC Comparison Pre and Post Distillation')  # 图形标题
plt.xticks(x, models, rotation=30)  # 设置x轴标签并倾斜30度
plt.legend()  # 显示图例

plt.tight_layout()  # 调整布局
plt.show()  # 显示图形
