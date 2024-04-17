import pandas as pd

# Creating data for models
data = {
    "Model": ["EEEA-Net-C2", "ofa-595", "OFA-595", "Visformer-small", "AutoformerV2-base"],
    "Parameters (M)": [4.73, 7.6, 6.9, 40.21, 71.12],
    "FLOPs (Billion)": [0.31, 0.54, 2.89, 4.75, 12.75]
}

# Create DataFrame
df = pd.DataFrame(data)

print(df)