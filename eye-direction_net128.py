# import tensorflow as tf
# from keras import backend as K
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list="0"
# sess = tf.Session(config=config)
# K.set_session(sess)
import tensorflow as tf
from keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import os, sys, time
import dlt
import numpy as np
from PIL import Image
from keras.utils import plot_model


# 変数設定
data_dir = './data/10faces/'
num_classes = 5
train_dirs = ['up', 'down', 'left', 'right', 'neutral']
test_dirs = ['upT', 'downT', 'leftT', 'rightT', 'neutralT']

SAVE_PATH = '10faces_20220616'

# データセット作成関数
def train_append(dir):
    X = []
    Y = []
    for i, d, in enumerate(dir):
        files = os.listdir(data_dir + d)
        for f in files:
            img = Image.open(data_dir + d + '/' + f, 'r')
            # ((左,上,右,下))
            imgC = img.crop((0, 34, 128, 66))
            # imgC = img.crop((0, 20, 128, 76))
            img_array = img_to_array(imgC)
            X.append(img_array)

            tmp = np.zeros(5)
            tmp[i] = 1
            Y.append(tmp)


    X = np.asarray(X)
    Y = np.asarray(Y)

    # X = X.reshape([-1, 56, 128, 3])
    X = X.reshape([-1, 32, 128, 3])

    return X, Y


(trainX, trainY) = train_append(train_dirs)
(testX, testY) = train_append(test_dirs)



# trainX = trainX.astype('float32') / 255  # RGBの値を0～1の値に正規化
# testX = testX.astype('float32') / 255

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


model = Sequential()  # インスタンスmodelを生成
# モデル

model.add(Conv2D(64, (2, 2), padding='same', input_shape=trainX.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# モデル構造の表示
# print(model.summary())
with open('results/modelsummary_' + SAVE_PATH + '.txt', 'w') as fp:
    model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

# コンパイル
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 過学習の抑制
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# 学習の実行
start = time.time()
fit = model.fit_generator(datagen.flow(trainX, trainY, batch_size=128),
                          steps_per_epoch=len(trainX) / 128,
                          validation_data=validationgen.flow(testX, testY),
                          epochs=7,
                          # callbacks=[early_stopping]
                          )
elapsed_time = time.time() - start
print('elapsed_time:{0}'.format(elapsed_time) + '[sec]')


# テストデータで評価
testX = testX.astype('float32') / 255  # RGBの値を0～1の値に正規化(学習時の形に合わせる)
score = model.evaluate(testX, testY, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# モデルの保存
folder = 'results'
if not os.path.exists(folder):
    os.makedirs(folder)

model.save(os.path.join(folder,  SAVE_PATH + '.h5'))

# 学習結果の確率と損失関数の確認
dlt.utils.plot_loss_and_accuracy(fit,  # model.fitのインスタンス
    fname=os.path.join('results', SAVE_PATH + '.png'))  # 保存するファイル名とパス


preds = model.predict(testX)  # 学習させたモデルから分類ラベルを取り出す
cls = model.predict_classes(testX)  # softmaxの出力するリストの最大値のラベルとして取り出す
print(cls)
