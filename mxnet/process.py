# import cv2
import os
# import numpy as np
list = os.listdir('./')
names = open('sn.txt', 'a')
for path in list:
    names.write('(\'mxnet\\\\' + path +'\' ,\'.\'),\n')