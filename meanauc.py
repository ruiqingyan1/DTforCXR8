# Manually calculating the average AUC for each model to handle potential data issues

import numpy as np



# Data for each model

wang_auc = np.array([0.716, 0.807, 0.784, 0.609, 0.706, 0.671, 0.633, 0.806, 0.708, 0.835, 0.815, 0.769, 0.708, 0.767])

yao_auc = np.array([0.772, 0.904, 0.859, 0.695, 0.792, 0.717, 0.713, 0.841, 0.788, 0.882, 0.829, 0.767, 0.765, 0.914])

chexnet_auc = np.array([0.8094, 0.9248, 0.8638, 0.7345, 0.8676, 0.7802, 0.768, 0.8887, 0.7901, 0.8878, 0.9371, 0.8047, 0.8062, 0.9164])

implemented_auc = np.array([0.8294, 0.9165, 0.887, 0.7143, 0.8597, 0.7873, 0.7745, 0.8726, 0.8142, 0.8932, 0.9254, 0.8304, 0.7831, 0.9104])

improved_auc = np.array([0.8311, 0.922, 0.8891, 0.7146, 0.8627, 0.7883, 0.782, 0.8844, 0.8148, 0.8992, 0.9343, 0.8385, 0.7914, 0.9206])
auc_values = [0.746, 0.888, 0.82, 0.684, 0.777, 0.717, 0.652, 0.817, 0.701, 0.837, 0.839, 0.777, 0.698, 0.787]
print(np.mean(auc_values))


# Calculate average AUC

average_aucs = {

    "Wang et al.": np.mean(wang_auc),

    "Yao et al.": np.mean(yao_auc),

    "CheXNet": np.mean(chexnet_auc),

    "Our Implemented CheXNet": np.mean(implemented_auc),

    "Our Improved Model": np.mean(improved_auc)

}
print(average_aucs)