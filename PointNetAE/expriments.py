import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pickle
import model
from tensorflow_graphics.nn.loss.chamfer_distance import evaluate

import h5py

# load data

with open(r'inputs\train_points.pkl', 'rb') as f:
    train = pickle.load(f)

with open(r'inputs\test_points.pkl', 'rb') as f:
    test = pickle.load(f)

model: keras.Model = model.build_model(2048, BATCH_SIZE=4, NUM_CLASSES=7)

# results = model.evaluate(X_test, X_test, batch_size=128)
# print("test loss, test acc:", results)

object_index = 500

index_list = [0, 100, 250, 500]

fig = plt.figure(figsize=(15, 10))

for i, index in enumerate(index_list):

    test_input = train[object_index]

    if i == 0:
        ax = fig.add_subplot(2, 4, 1, projection="3d")
        ax.scatter(test_input[:, 0], test_input[:, 1], test_input[:, 2], alpha=1)
        ax.set_axis_off()
        ax.title.set_text('test_input')

    model.load_weights(r'versions\cp-{0:04d}.ckpt'.format(index))

    test_pred = model.predict(train)

    test_output = test_pred[object_index]

    test_input = test_input.astype('double')

    test_output = test_output.astype('double')

    cd = evaluate(train[object_index].astype('double'), test_pred[object_index].astype('double'))

    ax = fig.add_subplot(2, 4, 2 + i, projection="3d")
    ax.scatter(test_output[:, 0], test_output[:, 1], test_output[:, 2], alpha=1)
    ax.set_axis_off()
    ax.title.set_text('epoch {0:04d} loss:{1:2.2e}'.format(index, cd))

plt.show()
