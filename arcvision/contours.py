'''Contains methods for finding contours in template'''

import cv2
import numpy as np

def find_template_contour(img):
    gray = cv2.bilateralFilter(img, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    _, contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:min(10, len(contours))]

    # for some reason, joining all seems to work ?
    c = np.concatenate( tuple(contours) )
    poly = cv2.approxPolyDP(c, 1, closed = True)

    # get polygon
    # add color back in so we can plot
    edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    hull = cv2.convexHull(poly)

    # now crop original image to new contour
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, (255,))
    img = cv2.bitwise_and(img, mask)

    return img, hull

def find_template_connected(img):
    gray = cv2.bilateralFilter(img, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    [num_labels, labels, stats, centroids] = cv2.connectedComponentsWithStats(edged, 8, cv2.CV_32S)
    # get the biggest connected
    label_ids = np.argsort(stats[cv2.CC_STAT_AREA,:], )[::-1]
    print(stats[cv2.CC_STAT_AREA,:])
    # add color back in so we can plot
    edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    for i in label_ids[:3]:
        pixels = np.argwhere(labels == i)
        poly = cv2.convexHull(pixels)
        cv2.polylines(edged, [poly], True, (0,0,255), 3)
    return edged, poly


def find_template_hull(img):
    gray = cv2.bilateralFilter(img, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    pixels = np.argwhere(edged > 0)

    #poly = cv2.convexHull(pixels)
    poly = cv2.approxPolyDP(pixels, 10, closed = True)
    edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    cv2.polylines(edged, [poly], True, (0,0,255), 3)
    return edged, poly