import numpy as np
import cv2

img = cv2.resize(cv2.imread('assets/cat.jpg', 0), (0, 0), fx=1.2, fy=1.2)
template = cv2.resize(cv2.imread('assets/template.jpg', 0), (0, 0), fx=0.8, fy=0.8)

img_edges = cv2.Canny(img, 50, 150)
template_edges = cv2.Canny(template, 50, 150)

h, w = template.shape

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img_edges.copy()

    result = cv2.matchTemplate(img2, template_edges, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    bottom_right = (location[0] + w, location[1] + h)    
    cv2.rectangle(img, location, bottom_right, (0,0,0), 5)
    cv2.imshow('Match', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


  