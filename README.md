# EyeGazeInput
畳み込みニューラルネットワークによって5方向の視線と瞬目(まばたき)を識別し、InputInterfaceの操作を行う視線入力プログラム。

## 実行環境
- pyuthon v3.7.4
- Keras v2.4.3
- opencv v4.2.0.34
- numpy v1.18.5
- Pillow v6.2.0
- matplotlib v3.1.1
- pyautogui v0.9.52
- pyperclip v1.8.1

## ファイル説明
- `EyeGazeInput.py`
    [InputInterface](https://github.com/22AMJ19/InputInterface.git)を呼び出し、操作する。
- `eye-direction_net128.py`
    畳み込みニューラルネットワークによって5方向の視線方向を学習した識別モデルを作成する。
- `eye-blink_net128.py`
    畳み込みニューラルネットワークによって瞬目（まばたき）を学習した識別モデルを作成する。
- `results`
    学習した視線方向と瞬目のモデルが入っているフォルダ。
- `haarcascade`
    OpenCVによる顔、目のHaar-like特徴分類器が入っているフォルダ。
    
## 使用方法
1. EyeGazeInput.pyを実行する。
2. 以下の画面が表示される。
![](https://i.imgur.com/5LfdizK.jpg)
3. 指定した判定時間同じ方向を注視することでフォーカスを操作する。
4. 判定時間目を閉じることで、選択の操作を行う。
5. 入力したい文字の子音にフォーカスされている状態で選択操作を行うことで以下の画面に遷移する。
![](https://i.imgur.com/OnmYmWT.jpg)
5. この画面で入力したい文字を選択することで文字が入力され、最初の画面に遷移する。
6. 「小,゛,゜」ボタンを選択することで、最後に入力されている文字の変換を行う。
7. 文字が入力されている状態で漢字ボタンを選択することで、予測変換がボタンに表示される。
![](https://i.imgur.com/EroW27Y.jpg)
8. この画面で変換したい文字を選択することで文字が変換され、最初の画面に遷移する。
9. 文字が入力されている状態で検索ボタンを選択することで、Google検索1ページ分のサイトのタイトルが表示されたボタンが表示される。
![](https://i.imgur.com/LuqYtDe.jpg)
10. この画面で開きたいサイトを選択することで、選択したサイトをブラウザで開く。
