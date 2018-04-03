from keras.layers import (
    Conv2D, BatchNormalization, Activation)

import tensorflow as tf


def ConvBNRelu(
        input_tensor,
        filters,
        kernel_size):

    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="valid")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x
