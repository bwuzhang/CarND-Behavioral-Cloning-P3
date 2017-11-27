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
