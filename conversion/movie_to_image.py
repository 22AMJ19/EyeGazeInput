import cv2
import os

def movie_to_image1(num_out, name):

    # name = '007'
    # date = '_20190827'
    # video_path = '../data/video4_20191028~/' + '16JK' + name + date + '.mp4'
    # output_path = '../data/image_from_video/10faces/' + '16JK' + name + '/'
    video_path = '../data/video/gaze20220616/' + name + '.mp4'
    output_path = '../data/image_from_video/10faces/' + name + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # キャプチャ動画の読み込み(キャプチャ構造体生成)
    capture = cv2.VideoCapture(video_path)

    img_count = 0  # 保存した候補が増数
    frame_count = 0  # 読み込んだフレーム画像数

    # フレーム画像がある限りループ
    while(capture.isOpened()):
        # フレーム画像を1枚取得
        ret, frame = capture.read()
        if ret == False:
            break

        # 指定した数だけフレーム画像を間引いて保存
        if frame_count % num_out == 0:
            img_file_name = output_path + str(img_count) + name + '.jpg'
            cv2.imwrite(img_file_name, frame)
            img_count += 1

        frame_count += 1

    # キャプチャ構造体開放
    capture.release()


def movie_to_image2(num_out):

    name = '007'
    date = '_20190827'
    video_path = '../data/video/video4_20191028~/' + '16JK' + name + date + '.mp4'
    output_path = '../data/image_from_video/10faces/' + '16JK' + name + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # キャプチャ動画の読み込み(キャプチャ構造体生成)
    capture = cv2.VideoCapture(video_path)

    img_count = 0  # 保存した候補が増数
    frame_count = 0  # 読み込んだフレーム画像数

    # フレーム画像がある限りループ
    while(capture.isOpened()):
        # フレーム画像を1枚取得
        ret, frame = capture.read()
        if ret == False:
            break

        # 指定した数だけフレーム画像を間引いて保存
        if frame_count % num_out == 0:
            img_file_name = output_path + str(img_count) + name + '.jpg'
            cv2.imwrite(img_file_name, frame)
            img_count += 1

        frame_count += 1

    # キャプチャ構造体開放
    capture.release()


if __name__ == '__main__':
    # 間引き数を設定してフレーム画像を抽出(3フレームごと)
    # ①ユーザーごと、方向ごとに一気に生成
    # names = ["16JK007", "16JK133", "16JK202", "16JK275", "17AJ002", "17AJ013", "17AJ085", "17AJ086", "17AJ097"]
    names = ["22amj19", "21amj13", "19aj001", "19aj015", "19aj094", "19aj109", "19aj127"]
    dirs = ["u", "d", "l", "r", "n"]
    for n in names:
        for d in dirs:
            video = n + d
            movie_to_image1(3, video)
            # print(video)

    # ②動画1個ずつ
    # movie_to_image2(3)



