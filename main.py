import cv2
import numpy as np
import masking
import edge
import surf

def main(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        skin = masking.skinMasking(frame)
        canny = edge.cannyEdge(skin)
        surfdetect = surf.featureDetection(canny)
        cv2.imshow('American Sign Language Recognizer', np.hstack([frame, skin]))
        cv2.imshow('Edge Detection', canny)
        cv2.imshow('Feature Detection', surfdetect)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
	main(0)
