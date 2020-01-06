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

# ======================================== #
#
#           Import Modules
#
# ======================================== #

# 公式モジュール
import cv2
import numpy as np
# from matplotlib import pyplot as plt
import time

# myself modules
import expansion_labeling_prcessing as ELP

print("----------Import Modules Clear----------")






# ======================================== #
#
#           Define Functions
#
# ======================================== #

print("----------Define Functions Clear----------")





# ======================================== #
#
#           Define GlobalValues
#
# ======================================== #

print("----------Define GlobalValues Clear----------")



# ======================================== #
#
#           Main code start
#
# ======================================== #

def main():
    """
        Description:
        ----------

        Parameter:
        ----------
        * None

        Return:
        ----------
        * None
    """




    # ===== Standart Histgram Data 読み込み===== #
    _img_hist_stand_red = cv2.imread("./hist_standard/R-1.jpg", 1)
    _img_hist_stand_red_hsv = cv2.cvtColor(_img_hist_stand_red, cv2.COLOR_BGR2HSV)
    _img_hist_stand_red_h = cv2.calcHist(_img_hist_stand_red_hsv, [0], None, [180], [0, 180])
    _img_hist_stand_red_h_w, _img_hist_stand_red_h_h, _img_hist_stand_red_h_ch = _img_hist_stand_red.shape[:3]

    # plt.hist(_img_hist_stand_red_h.ravel(), 256, [0, 256])
    # plt.hist(_img_hist_stand_red_h, 256, [0, 256])
    # plt.show()

    # cv2.namedWindow("hist", cv2.WINDOW_NORMAL)
    # cv2.imshow("hist", _img_hist_stand_red)



    # ===== 画像読み込み ===== #
    _img_src = cv2.imread("./img_data/temp1.jpg", 1)
    print("size={}".format(_img_src.shape))
    # 画像リサイズ
    _img_src = cv2.resize(_img_src, (1280, 720))

    while True:


        time_start = time.time()

        # cv2.namedWindow("_img_src", cv2.WINDOW_NORMAL)
        # cv2.imshow("_img_src", _img_src)


        # ===== グレイスケールへ変換 ===== #
        """
            一定しきい値を超える明るさの領域を探す
        """
        # BGR色空間からGrayへ変換
        _img_gray = cv2.cvtColor(_img_src, cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow("_img_gray", cv2.WINDOW_NORMAL)
        # cv2.imshow("_img_gray", _img_gray)
        
        ret, _img_thresh = cv2.threshold(_img_gray, 220, 255, cv2.THRESH_TOZERO)
        # cv2.namedWindow("_img_thresh", cv2.WINDOW_NORMAL)
        # cv2.imshow("_img_thresh", _img_thresh)
       



        # ===== Morphology ===== #
        _ERODE_KERNEL = np.ones((3, 3),np.uint8)
        """
        _img_erode_dst = cv2.erode(img_thresh, _ERODE_KERNEL, iterations = 1)    
        cv2.namedWindow("_img_erode_dst", cv2.WINDOW_NORMAL)
        cv2.imshow("_img_erode_dst", _img_erode_dst)
        """
        _img_opening_dst = cv2.morphologyEx(_img_thresh, cv2.MORPH_OPEN, _ERODE_KERNEL)
        cv2.namedWindow("_img_opening", cv2.WINDOW_NORMAL)
        cv2.imshow("_img_opening", _img_opening_dst)



        # ===== Labering処理 ===== # 
        """
            - 重心座標
            - サイズ
        """
        _debug_src = _img_src.copy()
       

        num, _data = ELP.expansion_labeling_prcessing(_img_opening_dst, [50, 2000], [0.0, 1.0], [False, False], _debug_src)
        print(num)
       




        
        # ===== HistgramInterseption ===== #
        """
            - Labering処理で得た明るい領域の位置をトリミングする.
            - 各領域において, 敵チームの色のヒストグラムとベースヒスグラムデータを照らしあわせて比較.
        """

        img_trim = _img_src.copy()
        img_hist_dst = _img_src.copy()

        img_armer_trim = [0 for i in range(0, num)]
        # print(img_armer_trim)

        _hist_object = np.zeros((num, 1), dtype=np.float64)

        hist_armer_trim = [0 for i in range(0, num)]

        for i in range(0, num):

            # triming
            img_armer_trim[i] = img_trim[int(_data[i][5]) : int(_data[i][5] + _data[i][7]) , int(_data[i][4]) : int(_data[i][4] + _data[i][6])]


            # resize
            img_armer_trim[i] = cv2.resize(img_armer_trim[i], (_img_hist_stand_red_h_w, _img_hist_stand_red_h_h))
            # print("hist w={}, h={}".format(_img_hist_stand_red_h_w, _img_hist_stand_red_h_h))
            # print("hist w={}, h={}".format(_img_hist_stand_red_h_w, _img_hist_stand_red_h_h))

            # convert to hsv
            img_armer_trim[i] = cv2.cvtColor(img_armer_trim[i], cv2.COLOR_BGR2HSV)

            # get hist
            hist_armer_trim[i] = cv2.calcHist(img_armer_trim[i], [0], None, [180], [0, 180])


            # comper hist
            _hist_object[i] = cv2.compareHist(hist_armer_trim[i], _img_hist_stand_red_h, cv2.HISTCMP_CORREL)
            print("detected object{}={}".format(i, _hist_object[i]))

            if(0.05 <= _hist_object[i]):
                img_hist_dst = cv2.rectangle(img_hist_dst, (int(_data[i][4]), int(_data[i][5])), (int(_data[i][4] + _data[i][6]), int(_data[i][5] + _data[i][7])), (0, 255, 0), 3)

            cv2.namedWindow("img_hist_dst", cv2.WINDOW_NORMAL)
            cv2.imshow("img_hist_dst", img_hist_dst)




        
        # ===== 装甲板を狙う処理 =====# 
        """
            - 装甲板についてる両側のLEDを見つけるため, 同じ形(アスペクト比とか)かつ, 一定ピクセル距離のものを見つける.
            - 条件を満たすものであれば, 真ん中の重心座標を狙う.
        """

        # ===== 相手の機体番号の取得 ===== #
        """
            - 装甲板の場所がわかれば, 装甲板のみをトリミングして, 番号を取得
                取得方法は, OCRか, テンプレートマッチングか(多分後者のほうが早そう)
        """




        # time.sleep(1.0)


        time_end = time.time()

        print("[ProcessTime] {}[s], {}[FPS]".format(time_end - time_start, 1 / (time_end - time_start)))

        # ===== key Event ===== #
        _key = cv2.waitKey(1)
        
        if(_key):
            if(_key == ord("q")):
                print("finish")
                break;

    cv2.destroyAllWindows()



# ======================================== #
#
#           Main code finish
#
# ======================================== #

print("----------Finish Program----------")

if __name__ == "__main__":
    main()
