import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from scipy import ndimage
from matplotlib.colors import ListedColormap, BoundaryNorm
from config import config

def clean_segmentation_mask_3d(pred_mask):
    """Post-processing to remove salt-and-pepper noise in 3D volumes."""
    cleaned_mask = pred_mask.copy()
    for c in [1, 2, 3]:
        binary_class_mask = (pred_mask == c)
        # 3D structure for morphology
        opened_mask = ndimage.binary_opening(binary_class_mask, structure=np.ones((3,3,3))).astype(np.int32)
        diff = (binary_class_mask ^ opened_mask)
        cleaned_mask[diff] = 0
    return cleaned_mask

def sliding_window_inference(model, input_volume, patch_size=(128, 128, 128), num_classes=4):
    """
    Chops the large (e.g. 240x240x155) volume into overlapping blocks, predicts,
    and intelligently stitches them back together by averaging the probabilities.
    """
    h, w, d, c = input_volume.shape
    ph, pw, pd_dim = patch_size
    
    # Pad if smaller than patch
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    pad_d = max(0, pd_dim - d)
    
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        input_volume = np.pad(input_volume, ((0, pad_h), (0, pad_w), (0, pad_d), (0, 0)), mode='constant')
        h, w, d, c = input_volume.shape
        
    prob_map = np.zeros((h, w, d, num_classes), dtype=np.float32)
    count_map = np.zeros((h, w, d, num_classes), dtype=np.float32)
    
    step_h, step_w, step_d = ph // 2, pw // 2, pd_dim // 2
    
    # Calculate step coordinates ensuring we cover the entire edge
    x_steps = list(range(0, h - ph + 1, step_h))
    if x_steps[-1] != h - ph: x_steps.append(h - ph)
    
    y_steps = list(range(0, w - pw + 1, step_w))
    if y_steps[-1] != w - pw: y_steps.append(w - pw)
        
    z_steps = list(range(0, d - pd_dim + 1, step_d))
    if z_steps[-1] != d - pd_dim: z_steps.append(d - pd_dim)
    
    # Perform sliding window
    for x in x_steps:
        for y in y_steps:
            for z in z_steps:
                patch = input_volume[x:x+ph, y:y+pw, z:z+pd_dim, :]
                patch_batch = np.expand_dims(patch, axis=0)
                pred = model.predict(patch_batch, verbose=0)[0]
                
                prob_map[x:x+ph, y:y+pw, z:z+pd_dim, :] += pred
                count_map[x:x+ph, y:y+pw, z:z+pd_dim, :] += 1.0
                
    final_prob = prob_map / count_map
    
    # Unpad if padded
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        final_prob = final_prob[:h-pad_h, :w-pad_w, :d-pad_d, :]
        
    return np.argmax(final_prob, axis=-1)

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ('loss', 'val_loss', 'Loss (GDL+Focal)'),
        ('dice_coef', 'val_dice_coef', 'Dice Coefficient'),
        ('iou', 'val_iou', 'Jaccard Index (IoU)'),
        ('accuracy', 'val_accuracy', 'Pixel Accuracy')
    ]
    for i, (train_m, val_m, title) in enumerate(metrics):
        ax = axes[i//2, i%2]
        if train_m in history.history:
            ax.plot(history.history[train_m], label='Train', color='#1f77b4', linewidth=2)
            ax.plot(history.history[val_m], label='Validation', color='#ff7f0e', linewidth=2, linestyle='--')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm):
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', annot_kws={"size": 20},
                xticklabels=config.CLASS_NAMES, 
                yticklabels=config.CLASS_NAMES)
    plt.title('Voxel-Level Confusion Matrix (Normalized)', fontweight='bold', fontsize=20)
    plt.ylabel('Ground Truth Class', fontweight='bold', fontsize=18)
    plt.xlabel('Predicted Class', fontweight='bold', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16, rotation=90)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18) 
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "confusion_matrix_3d.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_hd95(y_true, y_pred):
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0: return 0.0
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0: return 100.0 
    d_to_true = ndimage.distance_transform_edt(1 - y_true)
    d_to_pred = ndimage.distance_transform_edt(1 - y_pred)
    dist_p_to_t = d_to_true[y_pred.astype(bool)]
    dist_t_to_p = d_to_pred[y_true.astype(bool)]
    all_dists = np.concatenate([dist_p_to_t, dist_t_to_p])
    if len(all_dists) > 0: return np.percentile(all_dists, 95)
    else: return 0.0

def calculate_metrics_per_class(y_true, y_pred, num_classes):
    metrics = {}
    for c in range(num_classes):
        p = (y_pred == c)
        t = (y_true == c)
        TP = np.sum(p & t)
        FP = np.sum(p & ~t)
        FN = np.sum(~p & t)
        TN = np.sum(~p & ~t)
        smooth = 1e-6
        sensitivity = TP / (TP + FN + smooth)
        specificity = TN / (TN + FP + smooth)
        precision = TP / (TP + FP + smooth)
        accuracy = (TP + TN) / (TP + TN + FP + FN + smooth)
        dice = (2 * TP) / (2 * TP + FP + FN + smooth)
        jaccard = TP / (TP + FP + FN + smooth)
        hd95 = calculate_hd95(t, p)
        metrics[c] = {
            "Dice": dice, "Jaccard": jaccard, "Sensitivity": sensitivity,
            "Specificity": specificity, "Accuracy": accuracy, "Precision": precision,
            "Recall": sensitivity, "F1-Score": dice, "HD95": hd95
        }
        
    p_whole = (y_pred > 0)
    t_whole = (y_true > 0)
    TP_w = np.sum(p_whole & t_whole)
    FP_w = np.sum(p_whole & ~t_whole)
    FN_w = np.sum(~p_whole & t_whole)
    TN_w = np.sum(~p_whole & ~t_whole)
    
    sensitivity_w = TP_w / (TP_w + FN_w + smooth)
    specificity_w = TN_w / (TN_w + FP_w + smooth)
    precision_w = TP_w / (TP_w + FP_w + smooth)
    accuracy_w = (TP_w + TN_w) / (TP_w + TN_w + FP_w + FN_w + smooth)
    dice_w = (2 * TP_w) / (2 * TP_w + FP_w + FN_w + smooth)
    jaccard_w = TP_w / (TP_w + FP_w + FN_w + smooth)
    hd95_w = calculate_hd95(t_whole, p_whole)
    
    metrics["Entire Tumor"] = {
        "Dice": dice_w, "Jaccard": jaccard_w, "Sensitivity": sensitivity_w,
        "Specificity": specificity_w, "Accuracy": accuracy_w, "Precision": precision_w,
        "Recall": sensitivity_w, "F1-Score": dice_w, "HD95": hd95_w
    }
    return metrics

def save_metrics_to_csv(metrics_list):
    if not metrics_list: return pd.DataFrame()
    final_data = []
    categories = list(enumerate(config.CLASS_NAMES))
    categories.append(("Entire Tumor", "Entire Tumor"))
    
    for c_key, c_name in categories:
        class_metrics = [m[c_key] for m in metrics_list]
        avg_metrics = {}
        for key in class_metrics[0].keys():
            avg_metrics[key] = np.mean([item[key] for item in class_metrics])
        avg_metrics["Class"] = c_name
        final_data.append(avg_metrics)
        
    df = pd.DataFrame(final_data)
    cols = ["Class", "Dice", "Jaccard", "HD95", "Sensitivity", "Specificity", "Accuracy", "Precision", "Recall", "F1-Score"]
    df = df[cols]
    csv_path = os.path.join(config.RESULTS_DIR, "test_metrics_3d.csv")
    df.to_csv(csv_path, index=False)
    return df

def plot_metrics_summary(df):
    if df.empty: return
    plot_cols = [c for c in df.columns if c not in ['Class', 'HD95']]
    df_melt = df.melt(id_vars="Class", value_vars=plot_cols, var_name="Metric", value_name="Score")
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Class", palette="viridis")
    plt.title("3D Evaluation Metrics by Class & Whole Tumor", fontweight='bold')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "metrics_summary_3d.png"), dpi=300)
    plt.close()

def visualize_3d_multi_view(x_vol, y_true_vol, y_pred_vol, save_name="3d_inference.png", title_suffix=""):
    """
    Extracts middle slices across the Axial, Sagittal, and Coronal planes of a 3D volume.
    """
    coords = np.where(y_true_vol > 0)
    if len(coords[0]) > 0:
        cx, cy, cz = int(np.mean(coords[0])), int(np.mean(coords[1])), int(np.mean(coords[2]))
    else:
        cx, cy, cz = x_vol.shape[0]//2, x_vol.shape[1]//2, x_vol.shape[2]//2
        
    mask_colors = ['black', '#377eb8', '#4daf4a', '#e41a1c']
    cmap_mask = ListedColormap(mask_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap_mask.N)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'3D Multi-View Analysis {title_suffix}', fontsize=18, fontweight='bold', fontfamily='serif')
    
    slices = [
        ("Axial", x_vol[:, :, cz, 0], y_true_vol[:, :, cz], y_pred_vol[:, :, cz]),
        ("Coronal", x_vol[:, cy, :, 0], y_true_vol[:, cy, :], y_pred_vol[:, cy, :]),
        ("Sagittal", x_vol[cx, :, :, 0], y_true_vol[cx, :, :], y_pred_vol[cx, :, :])
    ]
    
    for i, (plane, x_show, yt_show, yp_show) in enumerate(slices):
        axes[i, 0].imshow(x_show, cmap='gray')
        if i==0: axes[i, 0].set_title("T1c Modality", fontweight='bold')
        axes[i, 0].set_ylabel(plane, fontweight='bold', fontsize=14)
        
        axes[i, 1].imshow(yt_show, cmap=cmap_mask, norm=norm)
        if i==0: axes[i, 1].set_title("Ground Truth", fontweight='bold')
        
        axes[i, 2].imshow(yp_show, cmap=cmap_mask, norm=norm)
        if i==0: axes[i, 2].set_title("Prediction", fontweight='bold')
        
        axes[i, 3].imshow(x_show, cmap='gray')
        axes[i, 3].imshow(yp_show, cmap=cmap_mask, norm=norm, alpha=0.4)
        if i==0: axes[i, 3].set_title("Overlay", fontweight='bold')
        
        for j in range(4):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            
    patches = [plt.Rectangle((0,0),1,1, color=mask_colors[i]) for i in range(4)]
    fig.legend(patches, config.CLASS_NAMES, loc='lower center', ncol=4, fontsize=14, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.10)
    plt.savefig(os.path.join(config.PLOTS_DIR, save_name), dpi=300)
    plt.close()