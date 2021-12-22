import cv2
import numpy as np

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def highlight_im(im, mask, threshold=0.5):
    threshold = int(threshold*255)
    mask = (mask*255).astype(np.uint8)
    ret, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        cv2.drawContours(im, contour, -1, (255,0,0), thickness = 1)

    b,g,r = cv2.split(im)

    r = cv2.add(b, 90, dst = r, mask = binary, dtype = cv2.CV_8U)

    return cv2.merge((b,g,r), im)