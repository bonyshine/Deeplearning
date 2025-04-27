import pandas as pd

# Manually create a DataFrame
data = {
    'Class': ['all', 'Ajwa', 'Medjool', 'Meneifi', 'Nabtat Ali', 'Shaishe', 'Sokari', 'Sugaey'],
    'Images': [100, 24, 10, 17, 9, 15, 18, 7],
    'Instances': [281, 143, 46, 17, 9, 15, 44, 7],
    'Precision (P)': [0.928, 0.934, 0.97, 0.919, 1.0, 0.854, 1.0, 0.816],
    'Recall (R)': [0.566, 0.0909, 0.152, 0.669, 0.88, 0.8, 0.369, 1.0],
    'mAP50': [0.662, 0.231, 0.232, 0.876, 0.984, 0.85, 0.484, 0.978],
    'mAP50-95': [0.507, 0.108, 0.193, 0.622, 0.761, 0.671, 0.393, 0.805],
}

df = pd.DataFrame(data)

# Display it as a nice styled table
df.style.background_gradient(cmap="YlGnBu").set_precision(3)
