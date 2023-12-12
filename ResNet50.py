# Import dependencies
import numpy as np
import os
from PIL import Image
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dataSplit import dataSplit

# Load and split data
inputData = "/tf/spectrograms/1_NOISY_N2048_H512"
labels = "/tf/labels.csv"
testSplit = 0.1

dataSplit(inputData, labels, testSplit)

training_dir = pathlib.Path("/tf/spectrograms/workdir/training").with_suffix('')
test_dir = pathlib.Path("/tf/spectrograms/workdir/test").with_suffix('')

test_count = len(list(test_dir.glob('*/*.png')))
train_count = len(list(training_dir.glob('*/*.png')))
validation_split = test_count / train_count

# Define parameters and create datasets
batch_size = 100
epochs = 10
img_height = 224
img_width = 224

print("\033[1mCreating training and validation datasets:\033[0m")
training_ds, validation_ds = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    validation_split=validation_split,
    subset="both",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True
)

print("\n\033[1mCreating test dataset:\033[0m")
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True
)

class_names = training_ds.class_names
print("\nNames of", str(len(class_names)), "classes:", class_names)

# Build and compile model
# Instantiate sequential model type:
model = keras.Sequential()

# Add pretrained ResNet50 model:
model.add(keras.applications.resnet50.ResNet50(
    include_top=False,
    weights=None,
    pooling="avg"
))

# Add output layer with sigmoid act. function for binary classification:
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.layers[0].trainable = False

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.TruePositives(),
        keras.metrics.FalsePositives(),
        keras.metrics.TrueNegatives(),
        keras.metrics.FalseNegatives()
    ]
)

model.summary()

# Model training
steps_per_epoch = len(training_ds) // epochs

t0 = time.time()

model.fit(
    training_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_ds,
    validation_steps=1,
    epochs=epochs
)

t1 = time.time()
dt = (t1 - t0)

# Model evaluation
results = model.evaluate(
    test_ds,
    return_dict=True
)

def f1score(p, r):
    f1 = 2 / ((1 / p) + (1 / r))
    return f1

print("---------------------------------------")
print('\033[1m' + "Model metrics:" + '\033[0m')
for i in results:
    print(i + ": " + str(results[i]))

print("---------------------------------------")
print("F1 Score: " + str(f1score(results['precision'], results['recall'])))
print("Time to train:", dt)
print("---------------------------------------")

tp, fp = results['true_positives'], results['false_positives']
fn, tn = results['false_negatives'], results['true_negatives']
cmx = np.array([[tp, fp], [fn, tn]], np.int32)

cmx_plot = sns.heatmap(
    cmx / np.sum(cmx),
    cmap='Blues',
    annot=True,
    fmt=".1%",
    linewidth=5,
    cbar=False,
    square=True,
    xticklabels=['Spoof (1)', 'Real (0)'],
    yticklabels=['Spoof (1)', 'Real (0)']
)

cmx_plot.set(xlabel="Actual", ylabel="Predicted")
