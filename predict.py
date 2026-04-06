import tensorflow as tf
import numpy as np
import nibabel as nib
import os
import argparse
from config import config
from data_preprocess import load_nifti
from losses import combined_loss, dice_coef
from model import ChannelAttention, SpatialAttention
import utils

def predict_single_scan_3d(scan_folder):
    model_path = os.path.join(config.CHECKPOINT_DIR, "att_unet_3d_best.keras")
    
    custom_objects = {
    "combined_loss": combined_loss, 
    "dice_coef": dice_coef,
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention
    }
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the 3D DAU-Net first.")
        return

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    scan_id = os.path.basename(os.path.normpath(scan_folder))
    print(f"Processing 3D Patient Volume: {scan_id}...")
    
    f_t1c = os.path.join(scan_folder, f"{scan_id}-t1c.nii.gz")
    f_t1n = os.path.join(scan_folder, f"{scan_id}-t1n.nii.gz")
    f_t2f = os.path.join(scan_folder, f"{scan_id}-t2f.nii.gz")
    f_t2w = os.path.join(scan_folder, f"{scan_id}-t2w.nii.gz")

    paths = [f_t1c, f_t1n, f_t2f, f_t2w]
    
    if not all(os.path.exists(p) for p in paths):
        print(f"Error: One or more NIfTI files missing in {scan_folder}")
        return

    try:
        images = [load_nifti(p) for p in paths]
        input_stack = np.stack(images, axis=-1) 
        
        print("  > Running 3D Sliding Window Inference (This may take a moment)...")
        patch_size = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_DEPTH)
        pred_mask_raw = utils.sliding_window_inference(model, input_stack, patch_size=patch_size)
        
        pred_mask_cleaned = utils.clean_segmentation_mask_3d(pred_mask_raw)

        # Save back as NIfTI to preserve physical spacing and headers
        out_path = os.path.join(config.RESULTS_DIR, f"{scan_id}_3D_segmentation_pred.nii.gz")
        original_nifti = nib.load(f_t1c)
        pred_nifti = nib.Nifti1Image(pred_mask_cleaned.astype(np.uint8), original_nifti.affine, original_nifti.header)
        nib.save(pred_nifti, out_path)
        
        print(f"    Saved 3D Segmentation Volue: {out_path}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, required=True, help="Path to the folder of the single scan containing .nii.gz files")
    args = parser.parse_args()
    
    predict_single_scan_3d(args.scan_dir)