import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from config import config
from data import get_test_scan_ids
from data_preprocess import process_path, load_nifti
from losses import combined_loss, dice_coef
from logger import get_logger
import utils
from model import ChannelAttention, SpatialAttention

log = get_logger("EVAL_3D")

TARGET_PATIENTS = [
    "BraTS-GLI-00001-000", 
    "BraTS-GLI-00013-000", 
]

def evaluate():
    log.info("--------------------------------------------------")
    log.info("        EVALUATION 3D DEEP DAU-NET                ")
    log.info("--------------------------------------------------")

    model_path = os.path.join(config.CHECKPOINT_DIR, "att_unet_3d_best.keras")
    if not os.path.exists(model_path):
        log.error("Model not found. Please train the 3D pipeline first.")
        return

    custom_objects = {
    "combined_loss": combined_loss, 
    "dice_coef": dice_coef,
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention
    }
    
    log.info(f"Loading 3D model from {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    test_ids = get_test_scan_ids()
    if not test_ids:
        log.error(f"No data found in {config.VAL_DIR}")
        return

    log.info(f"Found {len(test_ids)} patients. Processing full volumes...")
    
    batch_metrics_list = []
    total_cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES))
    
    for scan_id in tqdm(test_ids, desc="Evaluating Volumes"):
        try:
            t1c_p, t1n_p, t2f_p, t2w_p, seg_path = process_path(scan_id, config.VAL_DIR)
            
            # Load full spatial volumes
            gt_mask = load_nifti(seg_path, is_mask=True)
            img_t1c = load_nifti(t1c_p)
            img_t1n = load_nifti(t1n_p)
            img_t2f = load_nifti(t2f_p)
            img_t2w = load_nifti(t2w_p)
            
            input_stack = np.stack([img_t1c, img_t1n, img_t2f, img_t2w], axis=-1)
            
            # Run 3D Sliding Window Inference
            patch_size = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_DEPTH)
            y_pred_int = utils.sliding_window_inference(model, input_stack, patch_size=patch_size)
            
            # Post-Processing
            y_pred_cleaned = utils.clean_segmentation_mask_3d(y_pred_int)
            
            # Accumulate metrics
            y_true_flat = gt_mask.flatten()
            y_pred_flat = y_pred_cleaned.flatten()
            
            cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(config.NUM_CLASSES))
            total_cm += cm
            
            m = utils.calculate_metrics_per_class(gt_mask, y_pred_cleaned, config.NUM_CLASSES)
            batch_metrics_list.append(m)
            
            # Visualization for targeted subsets or all if short list
            if scan_id in TARGET_PATIENTS or len(test_ids) <= 5:
                save_name = f"{scan_id}_3D_MultiView.png"
                utils.visualize_3d_multi_view(
                    input_stack, 
                    gt_mask,
                    y_pred_cleaned,
                    title_suffix=f"({scan_id} - 3D DAU-Net)",
                    save_name=save_name
                )
                    
        except Exception as e:
            log.error(f"Error evaluating {scan_id}: {e}")
            continue

    log.info("Aggregating metrics...")
    df_metrics = utils.save_metrics_to_csv(batch_metrics_list)
    
    if df_metrics.empty:
        log.error("Evaluation produced no metrics.")
        return

    log.info("Generating Metrics and Confusion Matrix Plots...")
    utils.plot_metrics_summary(df_metrics)
    utils.plot_confusion_matrix(total_cm)
    
    print("\n" + "="*50)
    print("3D DAU-Net Performance Summary")
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("="*50 + "\n")
    
    log.info(f"Evaluation Complete. Artifacts saved in {config.OUTPUT_ROOT}.")

if __name__ == "__main__":
    evaluate()