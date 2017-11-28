import csv
import cv2
import numpy as np
import math
import os
import math

from find_lane import *

images = []

for file in os.listdir('examples'):
    if file.endswith('.jpg'):
        image = cv2.imread('examples/'+file)
        image_w_lines = find_lane(image)
        cv2.imwrite('examples/o_'+file,image_w_lines)

        image_flipped = cv2.flip(image_w_lines, 1)
        cv2.imwrite('examples/f_'+file,image_flipped)

        image_hsv = cv2.cvtColor(image_w_lines,cv2.COLOR_BGR2HSV)
        image_hsv = np.array(image_hsv, dtype = np.float64)
        random_bright = .5+np.random.uniform()
        image_hsv[:,:,2] = image_hsv[:,:,2]*random_bright
        image_hsv[:,:,2][image_hsv[:,:,2]>255]  = 255
        image_brightness = np.array(image_hsv, dtype = np.uint8)
        image_brightness = cv2.cvtColor(image_brightness,cv2.COLOR_HSV2BGR)
        cv2.imwrite('examples/b_'+file,image_brightness)

        image_shadow = np.array(image_w_lines, dtype = np.float64)
        h,w = image_shadow.shape[0:2]
        # mid = np.random.randint(0,w)
        mid = int(w/2)
        # factor = np.random.uniform(0.6,0.8)
        factor = 0.6
        if np.random.rand() > .5:
            image_shadow[:,0:mid,:] *= factor
        else:
            image_shadow[:,mid:w,:] *= factor
        image_shadow = np.array(image_shadow, dtype = np.uint8)
        cv2.imwrite('examples/s_'+file,image_shadow)

        image_shift = image_w_lines
        h,w,_ = image_shift.shape
        horizon = 2*h/5
        # v_shift = np.random.randint(-h/8,h/8)
        v_shift = -h/8
        pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
        pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        image_shift = cv2.warpPerspective(image_shift,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite('examples/sh_'+file,image_shift)
