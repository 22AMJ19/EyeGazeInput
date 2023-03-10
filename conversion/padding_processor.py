import numpy as np
import cv2
import os
import glob
from PIL import Image, ImageEnhance

names = ["22amj19", "21amj13", "19aj001", "19aj015", "19aj109", "19aj127"]
dirs = ["u", "d", "l", "r", "n"]
folders = []
for n in names:
    for d in dirs:
        folders.append(n + d)

savedir = ['up', 'down', 'left', 'right', 'neutral']
data = '10faces'


PN = 1  # up,downなどの大フォルダに入れる用のチェック
for d in folders:

    # IMG_INTER_PATHはxxxuA,xxxdA....のフォルダ(中間)
    # IMG_OUT_PATHはxxxuAA,xxxdAA....のフォルダ(単体保存用)
    # IMG_OUT_PATH2は実際に学習に使用するdata/10faces/up,down....のフォルダ
    IMG_PATH = '../data/' + data + '/org/' + d + '/'
    IMG_INTER_PATH = '../data/' + data + '/aug/' + d + 'A/'
    IMG_OUT_PATH = '../data/' + data + '/aug/' + d + 'AA/'
    i = PN % 5 - 1
    # print(savedir[i])
    IMG_OUT_PATH2 = '../data/' + data + '/' + savedir[i] + '/'
    if not os.path.exists(IMG_INTER_PATH):
        os.makedirs(IMG_INTER_PATH)
    if not os.path.exists(IMG_OUT_PATH):
        os.makedirs(IMG_OUT_PATH)
    if not os.path.exists(IMG_OUT_PATH2):
        os.makedirs(IMG_OUT_PATH2)

    # name = '_' + num + dir[i]

    no1 = 1
    files = glob.glob(IMG_PATH + '*.jpg')
    for f in files:
        name1, ext = os.path.splitext(os.path.basename(f))
        # print(name1)
        img = Image.open(f)

        # 彩度
        color_converter = ImageEnhance.Color(img)
        # コントラスト
        contrast_converter = ImageEnhance.Contrast(img)

        color_img1 = color_converter.enhance(1.3)
        color_img2 = color_converter.enhance(0.8)
        contrast_img1 = contrast_converter.enhance(1.5)
        contrast_img2 = contrast_converter.enhance(0.8)

        # IMG_INTER_PATH
        color_img1.save(IMG_INTER_PATH + name1 + '(' + str(no1) + ')' + 'col1' + '.jpg')
        color_img2.save(IMG_INTER_PATH + name1 + '(' + str(no1) + ')' + 'col2' + '.jpg')
        contrast_img1.save(IMG_INTER_PATH + name1 + '(' + str(no1) + ')' + 'con1' + '.jpg')
        contrast_img2.save(IMG_INTER_PATH + name1 + '(' + str(no1) + ')' + 'con2' + '.jpg')

        no1 += 1


    inter_files = glob.glob(IMG_INTER_PATH + '*.jpg')
    no2 = 1
    for f in inter_files:
        name2, ext = os.path.splitext(os.path.basename(f))
        img = Image.open(f)

        # 明度
        brightness_converter = ImageEnhance.Brightness(img)
        # シャープネス
        sharpness_converter = ImageEnhance.Sharpness(img)

        brightness_img1 = brightness_converter.enhance(0.8)
        brightness_img2 = brightness_converter.enhance(1.2)
        sharpness_img1 = sharpness_converter.enhance(0.5)
        sharpness_img2 = sharpness_converter.enhance(1.5)

        # IMG_OUT_PATHは学習に使用するxxxuAA,xxxdAA....のフォルダ
        brightness_img1.save(IMG_OUT_PATH + name2 + '(' + str(no2) + ')' + 'bri1' + '.jpg')
        brightness_img2.save(IMG_OUT_PATH + name2 + '(' + str(no2) + ')' + 'bri2' + '.jpg')
        sharpness_img1.save(IMG_OUT_PATH + name2 + '(' + str(no2) + ')' + 'sha1' + '.jpg')
        sharpness_img2.save(IMG_OUT_PATH + name2 + '(' + str(no2) + ')' + 'sha2' + '.jpg')

        # IMG_OUT_PATH2は学習に使用するup,down....のフォルダ
        brightness_img1.save(IMG_OUT_PATH2 + name2 + '(' + str(no2) + ')' + 'bri1' + '.jpg')
        brightness_img2.save(IMG_OUT_PATH2 + name2 + '(' + str(no2) + ')' + 'bri2' + '.jpg')
        sharpness_img1.save(IMG_OUT_PATH2 + name2 + '(' + str(no2) + ')' + 'sha1' + '.jpg')
        sharpness_img2.save(IMG_OUT_PATH2 + name2 + '(' + str(no2) + ')' + 'sha2' + '.jpg')

        no2 += 1

    PN += 1

