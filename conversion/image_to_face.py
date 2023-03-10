# import dlib
import cv2
import glob
import os
from PIL import Image

# 候補画像を格納したディレクトリパス

# glob mainグローバル変数ではフルパス

# id = ['007', '133', '140', '202', '275']
# dir = ['u', 'd', 'l', 'r', 'n']

# names = ["16JK007", "16JK133", "16JK202", "16JK275", "17AJ002", "17AJ013", "17AJ085", "17AJ086", "17AJ097"]
names = ["22amj19", "21amj13", "19aj001", "19aj015", "19aj094", "19aj109", "19aj127"]
dirs = ["u", "d", "l", "r", "n"]
folders = []
for n in names:
    for d in dirs:
        folders.append(n + d)

# name = '007u'
size = 128
# IMG_PATH = 'C:/Users/sugaya/PycharmProjects/keras/faces/data/image_from_video/5faces/' + '16JK' + id + dir + '/'
# IMG_OUT_PATH = '../data/5faces256/org/' + '16JK' + id + dir +  '/'
#
# if not os.path.exists(IMG_OUT_PATH):
#     os.makedirs(IMG_OUT_PATH)

# HaarCascade識別器が参照するXMLファイルパス
# XML_PATH_CV2 = '../haarcascade/haarcascade_frontalface_default.xml'
XML_PATH_CV2 = '../haarcascade/haarcascade_frontalface_alt.xml'


def face_detect_cv2(img_list, name):

    # 顔領域識別器をセット
    classifier = cv2.CascadeClassifier(XML_PATH_CV2)

    # 顔領域を検出すれば顔画像生成
    img_count = 1
    for img_path in img_list:

        # トリミング用カラー画像
        org_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # 識別器入力用グレースケール画像
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        #候補画像（グレースケール画像）を識別器に入力、顔領域と思われる座標情報取得
        face_points = classifier.detectMultiScale(gray_img,
                                                  scaleFactor=1.2,
                                                  minNeighbors=2,
                                                  minSize=(50, 50))

        #識別結果（矩形座標情報）よりカラー画像をトリミングして顔画像生成　
        for points in face_points:

            # 顔領域の座標点取得
            x, y, w, h = points

            # 顔領域トリミング
            dst_img = org_img[y:y+h, x:x+w]

            # 顔画像サイズ正規化して保存
            # face_img = cv2.resize(dst_img, (64, 64))
            face_img = cv2.resize(dst_img, (size, size))
            new_img_name = IMG_OUT_PATH + name + str(img_count) + '.jpg'
            cv2.imwrite(new_img_name, face_img)
            img_count += 1


def face_detect_dlib(img_list):

    # 顔領域を検出する識別器呼び出し
    detector = dlib.get_frontal_face_detector()

    # 候補画像を識別器にかけて座標情報を取得
    for img_path in img_list:

        # トリミング用候補画像（カラーモード）
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # 画像配列を組み直して識別器に入力、顔領域と思われる座標情報取得
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(cv_img, 1)

        # 顔領域を検出すれば顔画像生成
        img_count = 1
        for face in faces:

            # 候補画像サイズ取得
            height, width = img.shape[:2]

            # 顔領域の座標点取得
            top = face.top()
            bottom = face.bottom()
            left = face.left()
            right = face.right()

            # イレギュラーな顔領域は無視
            if not top < 0 and left < 0 and bottom > height and right > width:
                break
            else:
                # 顔領域トリミング
                dst_img = img[top:bottom, left:right]

                # 顔画像サイズ正規化して保存
                # face_img = cv2.resize(dst_img, (64, 64))
                face_img = cv2.resize(dst_img, (size, size))
                new_img_name = IMG_OUT_PATH + name + str(img_count) + '.jpg'
                cv2.imwrite(new_img_name, face_img)
                img_count += 1


for d in folders:
    # IMG_PATH = 'C:/Users/sugaya/PycharmProjects/keras/faces/data/image_from_video/10faces2/' + d + '/'
    IMG_PATH = '../data/image_from_video/10faces/' + d + '/'
    IMG_OUT_PATH = '../data/10faces/org/' + d + '/'
    if not os.path.exists(IMG_OUT_PATH):
        os.makedirs(IMG_OUT_PATH)

    #指定ディレクトリより候補画像を取得、API等を用いて顔画像生成
    images = glob.glob(IMG_PATH + '*.jpg')

    name = d + '_'
    face_detect_cv2(images, name)
    # face_detect_dlib(images)