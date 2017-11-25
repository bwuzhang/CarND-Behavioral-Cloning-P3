import csv
import cv2
import numpy as np
import math
import os
import math

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    img_shape = img.shape
    angle = np.array([])
    bias = np.array([])
    x1_list = np.array([])
    x2_list = np.array([])
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                x1_list = np.append(x1_list,[x1])
                x2_list = np.append(x2_list,[x2])
                angle = np.append(angle,[(y2-y1)/(x2-x1)])
                bias = np.append(bias,[y1-angle[-1]*x1])

    if len(angle[(angle<-0.6) & (angle>-0.85)]) > 0:
        left_lane_angle = angle[(angle<-0.6) & (angle>-0.85)].mean()
        left_lane_bias = bias[(angle<-0.6) & (angle>-0.85)].mean()
        left_lane_xmax = x2_list[(angle<-0.6) & (angle>-0.85)].max()
        left_x1 = int((img_shape[0]-left_lane_bias)/left_lane_angle)
        left_y1 = img_shape[0]
        left_x2 = int((img_shape[0] * 0.6-left_lane_bias)/left_lane_angle)
        left_y2 = int(img_shape[0] * 0.6)
        cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)
    
    if len(angle[(angle>0.5) & (angle<0.7)]) > 0:
        right_lane_angle = angle[(angle>0.5) & (angle<0.7)].mean()
        right_lane_bias = bias[(angle>0.5) & (angle<0.7)].mean()
        right_lane_xmin = x1_list[(angle>0.5) & (angle<0.7)].min()
        right_x1 = int((img_shape[0] * 0.6 - right_lane_bias)/right_lane_angle)
        right_y1 = int(img_shape[0] * 0.6)
        right_x2 = int((img_shape[0] - right_lane_bias)/right_lane_angle)
        right_y2 = img_shape[0]
        cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def find_lane(image):
    original_image = image
    image = gaussian_blur(image, 13)
    
    image = canny(image, 50, 70)
    
    image_shape = image.shape
    image = region_of_interest(image, np.array([[[0, 70],[320, 70],[320, 135],[0, 135]]]))
    
    lines = cv2.HoughLinesP(image, 1, np.pi/180, 50, np.array([]), minLineLength=35, maxLineGap=30)
    # draw_lines(original_image, lines, color=[255,0,0], thickness=5)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs((y1-y2)/(x1-x2)) > 0.1:
                    cv2.line(original_image, (x1, y1), (x2, y2), [0,0,0], 3)

    return original_image