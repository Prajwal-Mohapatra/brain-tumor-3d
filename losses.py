import tensorflow as tf
import tensorflow.keras.backend as K

def generalized_dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Generalized Dice Score 3D.
    Weights each class by the inverse of its volume squared.
    """
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    # Sum over spatial dimensions (Batch, H, W, D) -> Result shape: (Classes,)
    class_volumes = tf.reduce_sum(y_true_f, axis=[0, 1, 2, 3])
    weights = 1.0 / (class_volumes**2 + smooth)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=[0, 1, 2, 3])
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=[0, 1, 2, 3])
    
    numerator = tf.reduce_sum(weights * intersection)
    denominator = tf.reduce_sum(weights * union)
    
    return (2. * numerator + smooth) / (denominator + smooth)

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow((1 - y_pred), gamma)
    return K.mean(K.sum(weight * cross_entropy, axis=-1))

def combined_loss(y_true, y_pred):
    """Robust Loss: Generalized Dice + Focal"""
    return generalized_dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)