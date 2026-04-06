import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from config import config
from data_preprocess import data_generator

def get_scan_ids(root_dir):
    """Helper to get scan IDs dynamically by listing patient directories."""
    if not os.path.exists(root_dir):
        return []
    scan_ids = [d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d)) and "BraTS" in d]
    return sorted(scan_ids)

def get_train_val_datasets():
    """
    Loads data from dataset/TrainingData and splits it based on config.VAL_SPLIT (75/25).
    """
    print(f"Looking for Training data in: {config.TRAIN_DIR}")
    all_ids = get_scan_ids(config.TRAIN_DIR)
    
    if not all_ids:
        raise ValueError(f"No data found in {config.TRAIN_DIR}. Please check the path.")

    print(f"Found {len(all_ids)} scans. Generating training/val splits.")
    
    train_ids, val_ids = train_test_split(
        all_ids, 
        test_size=config.VAL_SPLIT, 
        random_state=config.SEED
    )
    
    # 3D TensorSpec
    output_signature = (
        tf.TensorSpec(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_DEPTH, config.NUM_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_DEPTH, config.NUM_CLASSES), dtype=tf.float32)
    )

    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_ids, config.TRAIN_DIR, is_train=True),
        output_signature=output_signature
    )

    val_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(val_ids, config.TRAIN_DIR, is_train=False),
        output_signature=output_signature
    )
    
    train_ds = train_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, len(train_ids), len(val_ids)

def get_test_scan_ids():
    """Returns the list of scans in the ValidationData directory for evaluation."""
    return get_scan_ids(config.VAL_DIR)