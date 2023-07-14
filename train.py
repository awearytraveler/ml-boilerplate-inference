import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
import os
import pickle
import argparse

import pathlib


def getDataStats(data_path):

    data_dir = pathlib.Path(data_path).with_suffix('')
    image_list = list(data_dir.glob('*/*.jpg'))
    widths = []
    heights = []

    for img in image_list:
        _temp = PIL.Image.open(str(img))
        widths.append(_temp.size[0])
        heights.append(_temp.size[1])

    AVG_HEIGHT = round(sum(heights)/len(heights))
    AVG_WIDTH = round(sum(widths)/len(widths))
    print(f"AVG_HEIGHT: {AVG_HEIGHT}\nAVG_WIDTH: {AVG_WIDTH}")
    return data_dir

def createDataset(data_path, img_height, img_width, batch_size=32):


    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    return train_ds, val_ds

def plotResults(history, epochs):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def vanillaModel(train, val, input_shape, epochs=50):

    # Model Definition
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=input_shape),
            layers.RandomRotation(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.BatchNormalization(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    epochs=50
    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs
        )
    
    plotResults(history=history, epochs=epochs)

if __name__=="__main__":

    img_height = 224
    img_width = 224
    base_learning_rate = 0.0001

    parser = argparse.ArgumentParser(description = "Training Module")
    parser.add_argument("-f", "--filePath", help = "Input File Path")

    args = parser.parse_args()

    data_dir = getDataStats(args.filePath)  # Eg argument - "/home/batknight/Downloads/FoodImage/FoodImage"
    train_ds, val_ds = createDataset(data_path=data_dir, img_height=img_height, img_width=img_width)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    hub_url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5'
    embed = hub.KerasLayer(hub_url)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(img_height, img_width, 3)),
            layers.RandomRotation(0.1),
        ]
    )

    model = tf.keras.Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        embed,
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    epochs=25
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
        )
    
    export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")
    tf.saved_model.save(model, export_module_dir)

    with open('labels.pickle', "wb") as f:
        f.write(pickle.dumps(class_names))

    
    plotResults(history=history, epochs=epochs)
    

    
