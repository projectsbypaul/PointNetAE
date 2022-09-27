import os
import glob
import pickle
import h5py
import tensorflow
import trimesh
import numpy as np
import tensorflow as tf
import h5df
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from model import build_model
from chamfer import get_loss
from keras.datasets import mnist


def loss_func(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)
    loss = get_loss(y_true, y_pred)
    return loss


NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

my_model = build_model(NUM_POINTS, BATCH_SIZE, NUM_CLASSES)

my_model.compile(
    loss=loss_func,
    optimizer=keras.optimizers.Adam(learning_rate=0.001))

train = tf.data.Dataset.load(r'/PythonComponents/segmentation/deep_learning/Datasets/ModelNet10/ae_train_data')

test = tf.data.Dataset.load(r'/PythonComponents/segmentation/deep_learning/Datasets/ModelNet10/ae_test_data')

my_model.fit(train, epochs=50, validation_data=test)



