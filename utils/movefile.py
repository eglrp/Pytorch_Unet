import cv2
import glob
import os
import shutil
Dir_img = '/home/buiduchanh/WorkSpace/Unet/Pytorch-UNet/data/data_rust/train_images'
Dir_anno = '/home/buiduchanh/WorkSpace/Unet/Pytorch-UNet/data/data_rust/small_anno'
Des_img = '/home/buiduchanh/WorkSpace/Unet/Pytorch-UNet/data/data_rust/small_image'
anolist = sorted(glob.glob('{}/*'.format(Dir_anno)))
for anno in anolist:
    print(anno)
    basename = os.path.splitext(os.path.basename(anno))[0]
    smallimage = os.path.join(Dir_img,basename + '.jpg')
    # print(smallimage)
    # exit()
    shutil.move(smallimage, Des_img)
