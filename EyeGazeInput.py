from keras.models import load_model
import os, sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#使用するファイルのインポート
#import scroll1
import pyautogui as pg
from time import sleep

#アプリ起動
import subprocess

# InputInterfaceのProject1.exeがあるパス
INTERFACE_PATH = "C:/Users/nidai/Source/Repos/22AMJ19/InputInterface/Project1/bin/Debug/Project1.exe" 

DECISION_TIME = 3 #フレーム数（判定時間）

#カウントの数を0とする
count = 3
b_count = 0
c_count = 0
flag =0

# 顔、目分類器の指定
face_cascade_file = "haarcascade/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_file)
eye_cascade_file = "haarcascade/haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(eye_cascade_file)

# モデルの設定
model_blink = load_model(os.path.join("results", "blink_test526.h5"))
model = load_model(os.path.join("results", "10faces_x50.h5"))
# model = load_model(os.path.join("results", "10faces_20220617.h5"))

def congaze(gazelist,n):
    j = -1
    if gazelist[-1]!=gazelist[-2]:
        return False
    for i in range(n-2):
        if gazelist[j-1:j]==gazelist[j-2:j-1]:
            j = j-1
        else:
            return False
    return True


def processing(gaze):
    global count,b_count,c_count
    global flag
    print(gaze)
    #視線方向推定の処理(↑、↓、←、→、正面の順)
    
    if b_count == 0:#アプリのセットアップ
        subprocess.Popen(INTERFACE_PATH) 
        print("待機中")
        sleep(5)
        b_count += 1
    elif b_count == 1:
        if gaze != "close":
            flag = 0
            print(str(c_count) + "あけ")
            if gaze == "up":
                pg.press('up')
            elif gaze == "down":
                pg.press('down')
            elif gaze == "left":
                pg.press('left')
            elif gaze == "right":
                pg.press('right')
            elif gaze == "neutral":
                # pg.click(xd, yd)
                print()
        elif gaze == "close":
            if flag == 0:
                print(str(c_count) + "とじ")
                pg.press('enter')
                flag = 1


def preprocessing(img, img_gray):
    # 指定部分の切り抜き
    img = cv2.resize(img, (128, 128), interpolation=cv2.IMREAD_COLOR)

    # 目を表示するウィンドウ
    EYE_WINDOW_NAME = "eye"
    cv2.namedWindow(EYE_WINDOW_NAME)

    eyes = ()  # タプル型の返却値 ... 判定しないとき
    # eyes = eye_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))  # 厳しめ
    # eyes = eye_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))  # ゆるめ

    if len(eyes) != 0:
        # ex = eyes[0][0]
        ey = eyes[0][1]
        # ew = eyes[0][2]
        eh = eyes[0][3]
        rg = int(((ey + eh / 2) / img_gray.shape[0]) * 128)  # 目の座標の縦軸の中心
        sys.stdout.write("目認識中! ")
        imgC = img[rg-16:rg+16, 0:128]
        imgR = cv2.resize(imgC, (160, 40))

        cv2.imshow(EYE_WINDOW_NAME, imgR)

        imgRGB = imgC[:, :, ::-1]
        img_reshape = np.reshape(imgRGB, (-1, 32, 128, 3))
        img_array = img_reshape.astype("float32") / 255
        return img_array

    # 目が認識されなかった場合(しない場合)
    sys.stdout.write("          ")
    imgC = img[34:66, 0:128]

    imgR = cv2.resize(imgC, (160, 40))
    cv2.imshow(EYE_WINDOW_NAME, imgR)

    imgRGB = imgC[:, :, ::-1]
    img_reshape = np.reshape(imgRGB, (-1, 32, 128, 3))
    img_array = img_reshape.astype("float32") / 255
    # print(img_reshape)
    # print(img_array)
    return img_array



if __name__ == "__main__":
    # 定数定義
    ESC_KEY = 27     # Escキー
    # E_KEY = 101      # Eキー
    INTERVAL = 11    # 待ち時間
    FRAME_RATE = 10  # fps
    DEVICE_ID = 0    # ウェブカメラの番号

    ORG_WINDOW_NAME = "org"
    FACE_WINDOW_NAME = "face"

    # カメラ映像取得
    # v_path = "../data/video/gaze20220616/22AMJ19l.mp4"
    # cap = cv2.VideoCapture(v_path) # 動画
    cap = cv2.VideoCapture(DEVICE_ID) # インカメラ
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)  # fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # 横幅540
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # 縦幅300

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(FACE_WINDOW_NAME)

    # 初期フレームの読み込み
    end_flag, c_frame = cap.read()
    gazelist = []

    # 変換処理ループ
    while end_flag is True:

        # 画像の取得と顔の検出
        
        pred = 0
        pred_blink = 0
        c_frame = cv2.resize(c_frame, (int(c_frame.shape[1] / 2), int(c_frame.shape[0] / 2)))
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=3,  minSize=(75, 75))
        # 検出した顔を囲む
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_gray = img_gray[y:y+h, x:x+w]
            cv2.rectangle(c_frame, (x-2, y-2), (x+w+4, y+h+4), (0, 0, 225), thickness=2)
            face_p = preprocessing(face, face_gray)  # 28x64に切り抜き、cvからPILへ形式変更
            pred = model.predict_classes(face_p)  # [数字]
            pred_blink = model_blink.predict_classes(face_p)

            face_resize = cv2.resize(face, (180, 180))
            blink = ["open", "close"]
            d = ["up", "down", "left", "right", "neutral"]
            for j in range(2):
                if pred_blink == j:
                    cv2.putText(face_resize, blink[j], (0, 30), cv2.FONT_HERSHEY_PLAIN,  2, (255, 255, 255), 2, cv2.LINE_AA)
                    # print(pred_blink, blink[j])
                    if blink[j] == "close":
                        gazelist.append(blink[j])
                    else:
                        for i in range(5):
                            if pred == i:
                                cv2.putText(face_resize, d[i], (0, 50), cv2.FONT_HERSHEY_PLAIN,  2, (255, 255, 255), 2, cv2.LINE_AA)
                                # print(pred, d[i])
                                gazelist.append(d[i])

            # 切り取った顔画像の表示
            cv2.imshow(FACE_WINDOW_NAME, face_resize)
        
        #視線入力による操作
        
        if len(gazelist)>DECISION_TIME-1 and congaze(gazelist, DECISION_TIME):
            processing(gazelist[-1])
            gazelist.clear()
        
        # キャプチャした画像の表示
        cv2.imshow(ORG_WINDOW_NAME, c_frame)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL) & 0XFF
        if key == ESC_KEY:
            break
        

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
