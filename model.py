import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, UpSampling3D, 
                                     Concatenate, Dropout, BatchNormalization, 
                                     Activation, GlobalAveragePooling3D, Reshape, 
                                     Dense, Multiply, Add, Layer)
from tensorflow.keras.models import Model
from config import config

# --- Custom 3D Layers ---

class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        filters = input_shape[-1]
        self.se_sq = GlobalAveragePooling3D()
        self.se_reshape = Reshape((1, 1, 1, filters))
        self.se_ex1 = Dense(filters // self.ratio, activation='relu', use_bias=False)
        self.se_ex2 = Dense(filters, activation='sigmoid', use_bias=False)
        self.multiply = Multiply()
        super(ChannelAttention, self).build(input_shape)

    def call(self, input_tensor):
        x = self.se_sq(input_tensor)
        x = self.se_reshape(x)
        x = self.se_ex1(x)
        x = self.se_ex2(x)
        return self.multiply([input_tensor, x])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({"ratio": self.ratio})
        return config

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.conv = Conv3D(1, (self.kernel_size, self.kernel_size, self.kernel_size), padding='same', activation='sigmoid')
        self.multiply = Multiply()
        self.concat = Concatenate(axis=-1)
        super(SpatialAttention, self).build(input_shape)

    def call(self, input_tensor):
        avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
        x = self.concat([avg_pool, max_pool])
        x = self.conv(x)
        return self.multiply([input_tensor, x])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

# --- Standard 3D U-Net Blocks ---

def conv_block(input_tensor, num_filters):
    x = Conv3D(num_filters, (3, 3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(num_filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters, dropout_rate=0.0):
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling3D((2, 2, 2))(x)
    if dropout_rate > 0:
        p = Dropout(dropout_rate)(p)
    return x, p

def dual_attention_block(input_tensor):
    ca = ChannelAttention()(input_tensor)
    sa = SpatialAttention()(input_tensor)
    x = Add()([input_tensor, ca, sa])
    return x

def decoder_block(input_tensor, skip_tensor, num_filters, dropout_rate=0.0, use_attention=True):
    x = UpSampling3D((2, 2, 2))(input_tensor)
    if use_attention:
        skip_tensor = dual_attention_block(skip_tensor)
    x = Concatenate()([x, skip_tensor])
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = conv_block(x, num_filters)
    return x

def build_unet_3d():
    inputs = Input((config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_DEPTH, config.NUM_CHANNELS))
    
    # Encoder
    c1, p1 = encoder_block(inputs, config.FILTERS, dropout_rate=0.0)
    c2, p2 = encoder_block(p1, config.FILTERS * 2, dropout_rate=0.1)
    c3, p3 = encoder_block(p2, config.FILTERS * 4, dropout_rate=0.2)
    c4, p4 = encoder_block(p3, config.FILTERS * 8, dropout_rate=0.2)
    
    # Bridge
    b = conv_block(p4, config.FILTERS * 16)
    
    # Decoder
    d4 = decoder_block(b, c4, config.FILTERS * 8, dropout_rate=0.2, use_attention=True)
    d3 = decoder_block(d4, c3, config.FILTERS * 4, dropout_rate=0.2, use_attention=True)
    d2 = decoder_block(d3, c2, config.FILTERS * 2, dropout_rate=0.1, use_attention=True)
    d1 = decoder_block(d2, c1, config.FILTERS, dropout_rate=0.0, use_attention=True)
    
    # Output
    outputs = Conv3D(config.NUM_CLASSES, (1, 1, 1), activation='softmax')(d1)
    
    model = Model(inputs=[inputs], outputs=[outputs], name="3D_DualAttention_UNet")
    return model

if __name__ == "__main__":
    model = build_unet_3d()
    model.summary()