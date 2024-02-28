import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def LPF(f_xy):
    # 2 次元高速フーリエ変換で周波数領域の情報を取り出す
    f_uv = np.fft.fft2(f_xy)
    # 画像の中心に低周波数の成分がくるように並べかえる
    shifted_f_uv = np.fft.fftshift(f_uv)

    # フィルタ (ローパス) を用意する
    x_pass_filter = Image.new(mode='L',  # 8-bit pixels, black and white
                              size=(shifted_f_uv.shape[1],
                                    shifted_f_uv.shape[0]),
                              color=0,  # default black
                              )
    # 中心に円を描く
    draw = ImageDraw.Draw(x_pass_filter)
    # 円の半径
    ellipse_r = 80
    # 画像の中心
    center = (shifted_f_uv.shape[1] // 2,
              shifted_f_uv.shape[0] // 2)
    # 円の座標
    ellipse_pos = (center[0] - ellipse_r,
                   center[1] - ellipse_r,
                   center[0] + ellipse_r,
                   center[1] + ellipse_r)
    draw.ellipse(ellipse_pos, fill=255)
    # フィルタ
    filter_array = np.asarray(x_pass_filter)

    # フィルタを適用する
    filtered_f_uv = np.multiply(shifted_f_uv, filter_array)

    # パワースペクトルに変換する
    magnitude_spectrum2d = 20 * np.log(np.absolute(filtered_f_uv))
    # plt.imshow(magnitude_spectrum2d)
    # plt.show()

    # 元の並びに直す
    unshifted_f_uv = np.fft.fftshift(filtered_f_uv)
    # 2 次元逆高速フーリエ変換で空間領域の情報に戻す
    i_f_xy = np.fft.ifft2(unshifted_f_uv).real  # 実数部だけ使う
    return i_f_xy


def tracking(filename: str):
    """動画から輪郭を抽出、軌跡を描画して新しい動画を作る関数

    :param filename: 元動画ファイル名.mp4
    """
    # 動画読み込み
    movie = cv2.VideoCapture(filename, 0)

    # 動画ファイル保存用の設定
    fps = int(movie.get(cv2.CAP_PROP_FPS))
    w = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # 元ファイルを引継ぎ保存するためのファイル名（元ファイル_out.元ファイルの拡張子）を生成
    name, ext = os.path.splitext(filename)
    out_path = name + '_out' + ext

    # 動画の仕様（保存するファイル名、fourcc, FPS, サイズ, カラー）
    video = cv2.VideoWriter(out_path, fourcc, fps, (w, h), True)

    ret, prev_frame = movie.read()  # フレームを取得
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    a = np.asarray(prev_frame)
    prev_fft = np.fft.fft2(a)

    i = 0

    while True:
        ret, frame = movie.read()  # フレームを取得
        if not ret: break # フレームが取得できない場合はループを抜ける

        # if i>10:
        #     i=0
        # else:
        #     i=i+1
        #     continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        a = np.asarray(frame)
        fft = np.fft.fft2(a)
        fft_diff = fft/prev_fft

        frame = LPF(np.asarray(frame))

        # plt.imshow(frame)
        # plt.show()
        img2 = prev_frame - frame

        prev_fft = fft
        prev_frame = frame

        # img2 = np.fft.ifft2(fft_diff).real

        img2 = img2 - np.min(img2)
        img2 = img2/np.max(img2)*255
        img2 = np.uint8(img2.real)

        # plt.imshow(np.abs(fft_diff))
        # plt.show()
        # plt.imshow(img2)
        # plt.show()
        video.write(cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR))

    # 動画オブジェクト解放
    movie.release()
    return

if __name__ == '__main__':
    filename = 'istockphoto-1323125363-640_adpp_is.mp4'
    filename = 'Calm Sea and Relaxing Sound of Waves [blAB_JqAJNw].f313.webm.part'
    tracking(filename)
