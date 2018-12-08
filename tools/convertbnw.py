import cv2
import glob
import os

Dir  = '/home/buiduchanh/WorkSpace/Unet/Pytorch-UNet/data/data_bridge/train_masks'
Des = '/home/buiduchanh/WorkSpace/Unet/Pytorch-UNet/data/data_bridge/train_masks_bw'
annolist = sorted(glob.glob('{}/*'.format(Dir)))
for img in annolist:
    print(img)
    basename = os.path.splitext(os.path.basename(img))[0]
    
    im_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    print(im_gray.shape)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    print(im_bw.shape)

    cv2.imwrite('test.jpg', im_bw)
    exit()

# img = '/home/buiduchanh/WorkSpace/Unet/Pytorch-UNet/data/data_bridge/train_masks_bw/610000251.jpg'
# img   = cv2.imread(img)
# print(img.shape)