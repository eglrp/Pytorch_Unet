import cv2
import numpy as np
import glob
import os

DIR = '/home/buiduchanh/WorkSpace/Unet/data/data_bridge/valid/original/rust_anno_val'
DES = '/home/buiduchanh/WorkSpace/Unet/data/data_bridge/valid/resized/valid_bw_images/bw_anno_bridge'
imglist = sorted(glob.glob('{}/*'.format(DIR)))
for img in imglist:
    basename = os.path.splitext(os.path.basename(img))[0]
    basename = basename[:-5]
    img = cv2.imread(img)
    print(basename)
    for i in range (len(img)):
        for j in range (len(img[i])):
            # print(img[i][j])
            B = img[i][j][0]
            G = img[i][j][1]
            R = img[i][j][2]
            if B > 200:
                img[i][j] = np.array([0, 0, 0])
            else:
                img[i][j] = np.array([255, 255, 255])

    cv2.imwrite('{}/{}.jpg'.format(DES,basename),img)
