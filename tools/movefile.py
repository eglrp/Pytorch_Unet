import cv2
import glob
import os
import shutil
from PIL import Image
Dir_anno = '/home/buiduchanh/WorkSpace/Unet/data/melona/test/annotations'
Des_anno = '/home/buiduchanh/WorkSpace/Unet/data/melona/test/annotation_tmp'
annolist = sorted(glob.glob('{}/*'.format(Dir_anno)))
for anno in annolist:
    basename = os.path.splitext(os.path.basename(anno))[0]
    img = Image.open(anno)
    img.save('{}/{}.jpg'.format(Des_anno,basename))

