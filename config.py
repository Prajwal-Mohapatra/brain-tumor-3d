import os
import matplotlib.pyplot as plt

class Config:
    # -- PATHS --
    BASE_DIR = "dataset"
    TRAIN_DIR = os.path.join(BASE_DIR, "TrainingData")
    VAL_DIR = os.path.join(BASE_DIR, "ValidationData")
    
    # -- OUTPUT STRUCTURE --
    OUTPUT_ROOT = "outputs_3d_pipeline"
    CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "models")
    LOGS_DIR = os.path.join(OUTPUT_ROOT, "logs")
    RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
    PLOTS_DIR = os.path.join(OUTPUT_ROOT, "plots")
    EXEC_LOG_FILE = os.path.join(LOGS_DIR, "execution.log")
    
    # -- 3D IMAGE SPECS --
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_DEPTH = 128
    NUM_CHANNELS = 4  # T1c, T1n, T2f, T2w
    NUM_CLASSES = 4   # 0: Background, 1: NCR, 2: ED, 3: ET
    CLASS_NAMES = ["Background", "NCR (Necrotic)", "ED (Edema)", "ET (Enhancing)"]
    
    # -- HYPERPARAMETERS --
    BATCH_SIZE = 2 # Reduced batch size to accommodate 3D patches in memory
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.25 # 75% Train / 25% Val as requested
    SEED = 42
    EARLY_STOPPING_PATIENCE = 15
    
    # Model Architecture
    FILTERS = 32      
    DROPOUT_RATE = 0.1
    
    def __init__(self):
        for d in [self.CHECKPOINT_DIR, self.LOGS_DIR, self.RESULTS_DIR, self.PLOTS_DIR]:
            os.makedirs(d, exist_ok=True)
            
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"],
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300
        })

config = Config()