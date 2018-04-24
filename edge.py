import numpy as np
import cv2

def cannyEdge(image, minVal=225, maxVal=250):
    """
    cannyEdge(image, sigma) inputs an image and sigma(default=0.33)
    and performs Canny edge detection using given mininum and maximum range 
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(image, minVal, maxVal)

    return edged
