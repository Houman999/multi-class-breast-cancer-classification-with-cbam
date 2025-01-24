# Library
import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, Multiply, Conv2D, Concatenate, Activation, Add

class CBAM(Layer):
    def __init__(self, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_dense_one = Dense(channel // self.reduction_ratio,
                                      activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.shared_dense_two = Dense(channel,
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.conv2d = Conv2D(filters=1,
                             kernel_size=7,
                             strides=1,
                             padding='same',
                             activation='sigmoid',
                             kernel_initializer='he_normal',
                             use_bias=False)

    def call(self, input_feature):
        # Channel Attention Module
        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, input_feature.shape[-1]))(avg_pool)
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, input_feature.shape[-1]))(max_pool)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        channel_attention = Activation('sigmoid')(Add()([avg_pool, max_pool]))
        channel_refined = Multiply()([input_feature, channel_attention])

        # Spatial Attention Module
        avg_pool = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = self.conv2d(concat)
        refined_feature = Multiply()([channel_refined, spatial_attention])

        return refined_feature