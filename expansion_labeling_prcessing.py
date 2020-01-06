#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Description:
    ----------

    Module used:
    ----------


    Author:
    ----------
        Shunya TANAKA

"""

# ========== Modules ========== #
import cv2
import os
import sys
import time
import numpy as np

# ========== Debugger ========== #
# Get File Name
FILE_NAME = os.path.basename(__file__)


def expansion_labeling_prcessing(img_bin, area_size, area_aspect, DEBUG, img_bgr):
    """

    Parameters
    ----------

    img_bin : image
        binary image

    area_size : list of float
        検出するプロブの面積の最小値, 最大値
        [0]: min : int
        [1]: max : int

    area_aspect : list of float
        検出オブジェクトのアスペクト比
        [0]: min 
        [1]: max 

    DEBUG : list of bool
        デバッグ機能有効化
        [0] : Terminal output 
        [1] : Window monitor output

    img_bgr : image
        DEBUGで使用するために必要

    Returns
    ----------

    _num_object : int
        フィルタを通過したプロブの数

    _list_data : list of float64
        フィルタを通過した各プロブのデータを格納したリスト


    """

    # Get Function Name
    FUNCTION_NAME = sys._getframe().f_code.co_name

    # Debugger Info
    DEBUG_INFO = "[" + FILE_NAME + ", " + FUNCTION_NAME + "]:"

    # ------------------------------ #

    # Labeling Process
    _num_object, _labeling, _contours, _GoCs = cv2.connectedComponentsWithStats(img_bin)

    # Define data list
    _list_data = np.zeros((_num_object, 9), dtype=np.float64)

    # 何もプロブが無いとエラーとなるので例外処理
    if (1 <= _num_object):

        # 条件にあうプロブを満たすものだけをカウント
        label = 0

        # プロブがあるとき, 0個目からデータを取得する
        for i in range(0, _num_object):

            # 重心座標取得
            center_x, center_y = _GoCs[i]

            # オブジェクトの左上の座標(x,y),横幅,縦幅,面積取得
            square_x, square_y, w, h, size = _contours[i]

            # Area filter & Aspect ratio filter
            if ((area_size[0] <= size) & (size <= area_size[1]) & (area_aspect[0] <= w / h) & (
                    w / h <= area_aspect[1])):

                _list_data[label][0] = label  # 割り当てられたラベルナンバー(あんまり使わない？)
                _list_data[label][1] = size  # ラベルナンバーの面積
                _list_data[label][2] = center_x  # 重心(x座標)
                _list_data[label][3] = center_y  # 重心(y座標)
                _list_data[label][4] = square_x  # 認識したオブジェクトの左上のx座標
                _list_data[label][5] = square_y  # 認識したオブジェクトの左上のx座標
                _list_data[label][6] = w  # 横幅
                _list_data[label][7] = h  # 縦幅
                _list_data[label][8] = w / h  # アスペクト比

                # DEBUG Monitor
                if (DEBUG[1] == True):
                    img_dst = cv2.rectangle(img_bgr, (int(_list_data[label][4]), int(_list_data[label][5])), (
                        int(_list_data[label][4] + _list_data[label][6]),
                        int(_list_data[label][5] + _list_data[label][7])), (0, 0, 255), 1)

                    img_dst = cv2.circle(img_bgr, (int(center_x), int(center_y)), 1, (0, 0, 255), -1)

                    img_dst = cv2.putText(img_dst, "{}".format(label), (int(center_x), int(center_y) - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.namedWindow(DEBUG_INFO + " result", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(DEBUG_INFO + " result", img_dst)

                # Debug Terminal
                if (DEBUG[0] == True):
                    print("-" * 30)
                    print(DEBUG_INFO)
                    print("label num        :{}".format(_list_data[label][0]))
                    print("size             :{}".format(_list_data[label][1]))
                    print("center_x         :{:.2f}".format(_list_data[label][2]))
                    print("center_y         :{:.2f}".format(_list_data[label][3]))
                    print("square_x         :{}".format(_list_data[label][4]))
                    print("square_y         :{}".format(_list_data[label][5]))
                    print("w                :{}".format(_list_data[label][6]))
                    print("h                :{}".format(_list_data[label][7]))
                    print("aspect           :{:.4f}".format(_list_data[label][8]))

                label += 1

            _num_object = label

    return _num_object, _list_data


if __name__ == "__main__":
    expansion_labeling_prcessing()