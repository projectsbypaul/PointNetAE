import os
import glob
import pickle
import h5py
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
import time
import sys


def loss_func(y_true, y_pred, batch_size):
    y_true = tf.cast(y_true, dtype='double')
    y_pred = tf.cast(y_pred, dtype='double')

    loss = evaluate(y_true, y_pred)

    loss = tf.reduce_mean(loss)

    return loss


# Hyperparam
NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE: int = 4
EPOCHS = 500
LEARNING_RATE = 0.001
BACKUP_INTERVAL = 10
DATASET = ['ModelNet10', ]
ARCHITECTURE = 'PointNet_AE'

# dataset

with open(r'inputs\train_points.pkl', 'rb') as f:
    train_points = pickle.load(f)

with open(r'inputs\test_points.pkl', 'rb') as f:
    test_points = pickle.load(f)

train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_points))
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_points))
test_dataset = test_dataset.batch(BATCH_SIZE)

# model and optimizer

model = build_model(NUM_POINTS, BATCH_SIZE, NUM_CLASSES)

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# set up logging

json_log = open(r'versions\loss_log.json', mode='wt', buffering=1)

# run training

for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch + 1))

    start_time = time.time()

    losses = []

    val_losses = []

    num_batches = train_dataset.cardinality().numpy()

    num_batches_val = test_dataset.cardinality().numpy()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.

        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_func(y_batch_train, logits, BATCH_SIZE)

            losses.append(loss_value)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        info_string_1 = "\rTraining loss (for one batch) at step {0:d}/{1:d}: {2:.4f} ".format(step + 1, num_batches,
                                                                                               loss_value)
        sys.stdout.write(info_string_1)
        sys.stdout.flush()

    epoch_loss = sum(losses) / num_batches

    epoch_loss = epoch_loss.numpy()

    info_string_2 = "\nTraining loss for epoch {0} : {1:.4f} ".format(epoch + 1, epoch_loss)
    print(info_string_2)

    # Validation loss

    for (x_batch_test, y_batch_test) in test_dataset:
        logits = model(x_batch_test, training=True)
        val_losses.append(loss_func(y_batch_test, logits, BATCH_SIZE))

    epoch_val_loss = sum(val_losses) / num_batches_val

    epoch_val_loss = epoch_val_loss.numpy()

    print("Validation Loss: %.4f" % (float(epoch_val_loss)))
    print("Time taken: %.2fs" % (time.time() - start_time))

    # log with json

    json_log.write(
        json.dumps({'epoch': epoch + 1, 'loss': epoch_loss, 'val_loss': epoch_val_loss})+ '\n')

    # save weights

    if epoch % BACKUP_INTERVAL == 0:
        model.save_weights(r'versions\cp-{0:04d}.h5'.format(epoch + 1))
