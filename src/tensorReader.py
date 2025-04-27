import pandas as pd
import matplotlib.pyplot as plt
# Step 4: Create subplots for comparing multiple metrics (now includes all 3 models)
fig, axs = plt.subplots(3, 3, figsize=(16, 13))
# Step 1: Load the result CSV files for both models
LayerCbam = pd.read_csv(r"runs\detect\train85_cbam1layer\results.csv")  # Replace with the path to your model 1 results CSV
model11vn = pd.read_csv(r"runs\detect\yolo11vn\results.csv")  # Replace with the path to your model 2 results CSV
layer2cbam= pd.read_csv(r"runs\detect\train89\results.csv")
# Step 2: Extract relevant columns for both models
model_1_metrics =model11vn [['epoch', 'train/box_loss', 'train/cls_loss', 
                               'metrics/precision(B)', 'metrics/recall(B)', 
                               'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 
                               'val/box_loss', 'val/cls_loss', 'val/dfl_loss']]

cbaml1_2_metrics =LayerCbam [['epoch', 'train/box_loss', 'train/cls_loss', 
                               'metrics/precision(B)', 'metrics/recall(B)', 
                               'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 
                               'val/box_loss', 'val/cls_loss', 'val/dfl_loss']]
cabm2_3_metrics =layer2cbam[['epoch', 'train/box_loss', 'train/cls_loss', 
                               'metrics/precision(B)', 'metrics/recall(B)', 
                               'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 
                               'val/box_loss', 'val/cls_loss', 'val/dfl_loss']]

# Step 3: Merge the two models' metrics on epoch
# First merge model 1 and model 2

cabm2_3_metrics = cabm2_3_metrics.rename(columns={
    col: f"{col}_cbam_2" for col in cabm2_3_metrics.columns if col != "epoch"
})

# Now do the merges
merged_df = pd.merge(cbaml1_2_metrics, model_1_metrics, on='epoch', suffixes=('_cbam_1', '_model_1'))
merged_df = pd.merge(merged_df, cabm2_3_metrics, on='epoch')
merged_df = pd.merge(cbaml1_2_metrics,model_1_metrics, on='epoch', suffixes=('_cbam_1', '_model_1'))
merged_df = pd.merge(merged_df, cabm2_3_metrics, on='epoch', suffixes=('', '_cbam_2'))


print("Columns in merged_df:")
print(merged_df.columns.tolist())

# Define line colors
colors = ['blue', 'red', 'green']

# Define model labels
labels = ['CBAM 1', 'Model 1', 'CBAM 2']

# Plot helper function
def plot_metric(ax, metric_name, title, ylabel):
    ax.plot(merged_df['epoch'], merged_df[f'{metric_name}_cbam_1'], label=labels[0], color=colors[0])
    ax.plot(merged_df['epoch'], merged_df[f'{metric_name}_model_1'], label=labels[1], color=colors[1])
    ax.plot(merged_df['epoch'], merged_df[f'{metric_name}_cbam_2'], label=labels[2], color=colors[2])
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend()

# First row
plot_metric(axs[0, 0], 'train/box_loss', 'Train Box Loss', 'Box Loss')
plot_metric(axs[0, 1], 'train/cls_loss', 'Train Class Loss', 'Class Loss')
plot_metric(axs[0, 2], 'metrics/precision(B)', 'Precision (B)', 'Precision')

# Second row
plot_metric(axs[1, 0], 'metrics/recall(B)', 'Recall (B)', 'Recall')
plot_metric(axs[1, 1], 'metrics/mAP50(B)', 'mAP50 (B)', 'mAP50')
plot_metric(axs[1, 2], 'metrics/mAP50-95(B)', 'mAP50-95 (B)', 'mAP50-95')

# Third row
plot_metric(axs[2, 0], 'val/box_loss', 'Validation Box Loss', 'Box Loss')
plot_metric(axs[2, 1], 'val/cls_loss', 'Validation Class Loss', 'Class Loss')
plot_metric(axs[2, 2], 'val/dfl_loss', 'Validation DFL Loss', 'DFL Loss')

plt.tight_layout()

# Step 5: Calculate overall metrics (3 models separately)
avg_metrics = {
    "CBAM 1": {
        "mAP50": merged_df['metrics/mAP50(B)_cbam_1'].mean(),
        "mAP50-95": merged_df['metrics/mAP50-95(B)_cbam_1'].mean(),
        "train_cls_loss": merged_df['train/cls_loss_cbam_1'].mean(),
        "val_cls_loss": merged_df['val/cls_loss_cbam_1'].mean()
    },
    "Model 1": {
        "mAP50": merged_df['metrics/mAP50(B)_model_1'].mean(),
        "mAP50-95": merged_df['metrics/mAP50-95(B)_model_1'].mean(),
        "train_cls_loss": merged_df['train/cls_loss_model_1'].mean(),
        "val_cls_loss": merged_df['val/cls_loss_model_1'].mean()
    },
    "CBAM 2": {
        "mAP50": merged_df['metrics/mAP50(B)_cbam_2'].mean(),
        "mAP50-95": merged_df['metrics/mAP50-95(B)_cbam_2'].mean(),
        "train_cls_loss": merged_df['train/cls_loss_cbam_2'].mean(),
        "val_cls_loss": merged_df['val/cls_loss_cbam_2'].mean()
    }
}

# Step 6: Print overall metrics
print("\n=== Overall Metrics ===")
for model_name, metrics in avg_metrics.items():
    print(f"{model_name}:")
    print(f"  Avg mAP50: {metrics['mAP50']:.4f}")
    print(f"  Avg mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Avg Train Class Loss: {metrics['train_cls_loss']:.4f}")
    print(f"  Avg Val Class Loss: {metrics['val_cls_loss']:.4f}\n")

# Show the plots
plt.show()
