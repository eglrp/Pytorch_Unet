import matplotlib.pyplot as plt
import glob
import os
import cv2


def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()

Dir_image = '/home/buiduchanh/WorkSpace/Unet/Pytorch-UNet/data/melona_test2018'
Dir_mask = '/home/buiduchanh/WorkSpace/Unet/Pytorch-UNet/data/result_melona'
imglist = sorted(glob.glob('{}/*'.format(Dir_image)))
for img in imglist:
    basename = os.path.splitext(os.path.basename(img))[0]
    result_name = basename + '_result.png'
    resultpath = os.path.join(Dir_mask, result_name)
    # plot_img_and_mask(cv2.imread(img, cv2.COLOR_RGB2BGR), cv2.imread(resultpath,  cv2.COLOR_RGB2BGR))
    infoimg = cv2.imread(img)
    informask = cv2.imread(resultpath)
    mask = informask[:,:,1]
    for i in range(len(infoimg)):
        for j in range(len(infoimg[i])):
            if mask[i][j] > 0:
                infoimg[i][j][1] = infoimg[i][j][1] + 50

    fig = plt.figure()
    # fig.set_title('Input image')
    plt.imshow(infoimg)
    plt.show()