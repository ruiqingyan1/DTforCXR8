import torch
import torch.nn.functional as F


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
