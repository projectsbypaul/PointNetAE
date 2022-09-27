import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt


def build_model(NUM_POINTS, BATCH_SIZE, NUM_CLASSES):
    inputs = keras.Input(shape=(NUM_POINTS, 3))

    x = layers.Reshape((1, NUM_POINTS * 3))(inputs)

    x = layers.Dense(3 * NUM_POINTS)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)

    encoded = layers.Dense(128)(x)

    # Decoder

    codeword = layers.Dense(128)(encoded)
    x = layers.BatchNormalization()(codeword)
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(NUM_POINTS * 3)(x)

    outputs = layers.Reshape((NUM_POINTS, 3))(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

    return model


if __name__ == '__main__':
    ae = build_model(2048, 4, 10)

    ae.summary()
