import numpy as np
import cv2

def skinMasking(image):
    """
    skinMasking(image) inputs an image and returns
    skin mask of the same using set range values

    This makes the function behave differently to different
    skin tones
    """
    lower = np.array([0, 48, 0], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")

    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(image, image, mask = skinMask)

    return skin
