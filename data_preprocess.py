import tensorflow as tf
import nibabel as nib
import numpy as np
import os
import random
from config import config

def normalize_volume(volume):
    """Z-score normalization ignoring the background (0)."""
    mask = volume > 0
    if mask.any():
        mean = volume[mask].mean()
        std = volume[mask].std()
        volume[mask] = (volume[mask] - mean) / (std + 1e-8)
    return volume

def load_nifti(path, is_mask=False):
    """Loads a NIfTI volume and applies normalization or label mapping."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    
    img = nib.load(path).get_fdata()
    
    if is_mask:
        img = np.round(img).astype(np.int32)
        # BRaTS masks are 0, 1, 2, 4. Map 4 to 3 for contiguous one-hot encoding.
        img[img == 4] = 3
        img = np.clip(img, 0, config.NUM_CLASSES - 1)
        return img
    else:
        return normalize_volume(img.astype(np.float32))

def process_path(scan_id, root_dir):
    """Constructs NIfTI file paths."""
    scan_dir = os.path.join(root_dir, scan_id)
    p_t1c = os.path.join(scan_dir, f"{scan_id}-t1c.nii.gz")
    p_t1n = os.path.join(scan_dir, f"{scan_id}-t1n.nii.gz")
    p_t2f = os.path.join(scan_dir, f"{scan_id}-t2f.nii.gz")
    p_t2w = os.path.join(scan_dir, f"{scan_id}-t2w.nii.gz")
    p_seg = os.path.join(scan_dir, f"{scan_id}-seg.nii.gz")
    return p_t1c, p_t1n, p_t2f, p_t2w, p_seg

def extract_random_patch(image, mask, patch_size):
    """Extracts a random 3D patch from the full volume to fit in GPU memory."""
    h, w, d, _ = image.shape
    ph, pw, pd = patch_size
    
    # Pad if volume is smaller than the requested patch
    if h < ph or w < pw or d < pd:
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        pad_d = max(0, pd - d)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, pad_d), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
        h, w, d = image.shape[:3]
        
    x = random.randint(0, h - ph)
    y = random.randint(0, w - pw)
    z = random.randint(0, d - pd)
    
    return image[x:x+ph, y:y+pw, z:z+pd, :], mask[x:x+ph, y:y+pw, z:z+pd]

def augment_data_3d(X, Y):
    """Random 3D flips across the three spatial axes."""
    if random.random() < 0.5:
        X = np.flip(X, axis=0)
        Y = np.flip(Y, axis=0)
    if random.random() < 0.5:
        X = np.flip(X, axis=1)
        Y = np.flip(Y, axis=1)
    if random.random() < 0.5:
        X = np.flip(X, axis=2)
        Y = np.flip(Y, axis=2)
    return X, Y

def data_generator(scan_ids, root_dir, is_train=True):
    """3D Data Generator that extracts patches on-the-fly."""
    patch_size = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_DEPTH)
    
    for scan_id in scan_ids:
        try:
            t1c_p, t1n_p, t2f_p, t2w_p, seg_p = process_path(scan_id, root_dir)
            
            # Load Mask
            mask = load_nifti(seg_p, is_mask=True)
            
            # Load Modalities and Stack
            img_t1c = load_nifti(t1c_p)
            img_t1n = load_nifti(t1n_p)
            img_t2f = load_nifti(t2f_p)
            img_t2w = load_nifti(t2w_p)
            
            X_raw = np.stack([img_t1c, img_t1n, img_t2f, img_t2w], axis=-1)
            Y_raw = mask
            
            # Extract a fixed-size patch
            X_patch, Y_patch = extract_random_patch(X_raw, Y_raw, patch_size)
            
            # One-hot encode the patch
            Y_patch_oh = tf.one_hot(Y_patch, depth=config.NUM_CLASSES).numpy()
            
            if is_train:
                X_patch, Y_patch_oh = augment_data_3d(X_patch, Y_patch_oh)
                
            yield X_patch, Y_patch_oh
            
        except Exception as e:
            continue