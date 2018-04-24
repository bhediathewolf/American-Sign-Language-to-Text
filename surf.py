import numpy as np
import cv2
from matplotlib import pyplot as plt

def featureDetection(canny):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(canny, None)
    srf = cv2.drawKeypoints(canny, kp, None, (0, 0, 255), 4)

    return srf
