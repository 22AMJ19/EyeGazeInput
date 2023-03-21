# vit keras v0.1.0
# tensor flow v2.10.1
# tensorflow-addons v0.19.0

import os, sys, time
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model, Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam
from vit_keras import vit


num_classes = 2 # 分類するクラス数

DATA_DIR = './data/blink/'

TRAIN_DIRS = ['train/open', 'train/close']
TEST_DIRS = ['val/openT', 'val/closeT']

# データセット作成関数
def train_append(dir):
    X = []
    Y = []
    for i, d, in enumerate(dir):
        files = os.listdir(DATA_DIR + d)
        for f in files:
            img = Image.open(DATA_DIR + d + '/' + f, 'r')
            # ((左,上,右,下))
            imgC = img.crop((0, 34, 128, 66)) # 顔画像から目の部分を切り抜き
            imgC = imgC.resize((128, 128)) # 128×128に拡大
            img_array = img_to_array(imgC)
            # img_array = img_to_array(img)
            X.append(img_array)

            tmp = np.zeros(num_classes)
            tmp[i] = 1
            Y.append(tmp)
            

    X = np.asarray(X)
    Y = np.asarray(Y)

    # X = X.reshape([-1, 32, 128, 3]) # -1サンプル数 32高さ 128幅 3RGBカラーイメージ(1だとグレイスケール)
    X = X.reshape([-1, 128, 128, 3])

    return X, Y

(x_train, y_train) = train_append(TRAIN_DIRS)
(x_test, y_test) = train_append(TEST_DIRS)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

image_size = 128

# Data Augmentation
datagen = ImageDataGenerator(rescale=1./255,  # RGBの値を0～1の値に正規化
                             rotation_range=5,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             channel_shift_range=80,
                             zoom_range=0.05)

validationgen = ImageDataGenerator(
    rescale=1./255   # RGBの値を0～1の値に正規化
    )

def buildModel():
    vit_model = vit.vit_b16(
        image_size = image_size,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        )

    model = Sequential()
    model.add(vit_model)
    model.add(Dense(num_classes, 'softmax'))

    # model.compile(optimizer=tfa.optimizers.RectifiedAdam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=tfa.optimizers.RectifiedAdam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

model = buildModel()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, 
                                                  patience=5, mode='auto')

file_path='weights.{epoch:02d}-val_loss.{val_loss:.2f}.h5'
check_point = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_accuracy',save_best_only=True, 
                                                 save_weights_only=True, period=1, mode='auto')

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, 
                                                    verbose=0, mode='auto', min_lr=1e-6)
epochs = 1000

history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    steps_per_epoch=len(x_train) / 128,
                    epochs = epochs,
                    validation_data=validationgen.flow(x_test, y_test),
                    shuffle=True,
                    callbacks=[early_stopping, check_point, lr_scheduler])

model.save("results.h5")