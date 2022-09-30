import os
import glob
import pickle
import h5py
import tensorflow
import trimesh
import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import json
from tensorflow import keras
from matplotlib import pyplot as plt
from model import build_model
from tensorflow_graphics.nn.loss.chamfer_distance import evaluate
from keras.callbacks import LambdaCallback


def loss_func(y_true, y_pred):
    loss = evaluate(y_true, y_pred)
    return loss


# Hyperparam
NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE: int = 4
EPOCHS = 500
LEARNING_RATE = 0.001
DATASET = ['ModelNet10', ]
ARCHITECTURE = 'PointNet_AE'

# checkpoint callback
checkpoint_path = r"versions/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every epoch: period = 1 .
    period=10)

# json log callback
json_log = open('versions/loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss'], 'val_loss': logs['val_loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

# compile and train

my_model = build_model(NUM_POINTS, BATCH_SIZE, NUM_CLASSES)

my_model.compile(
    loss=loss_func,
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

with open(r'inputs\train_points.pkl', 'rb') as f:
    train = pickle.load(f)

with open(r'inputs\test_points.pkl', 'rb') as f:
    test = pickle.load(f)

my_model.save_weights(checkpoint_path.format(epoch=0))

my_model.fit(train, train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_data=(test, test),
             callbacks=[cp_callback, json_logging_callback])
