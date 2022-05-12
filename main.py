import cv2
import numpy as np
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector


def alphaMerge(img, icon, top, left):
    part_of_fg = icon.copy()
    icon = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    icon[icon!=255] = 0         

    part_of_fg[part_of_fg==255] = 0
    part_of_fg = cv2.erode(part_of_fg, np.ones((7, 7)))
    part_of_fg = cv2.dilate(part_of_fg, np.ones((3, 3)))
    # part_of_fg = cv2.bitwise_not(part_of_fg)

    # rows, cols, _ = part_of_bg.shape
    # M = cv2.getRotationMatrix2D((rows/2, cols/2), -45, 1)
    # part_of_bg = cv2.warpAffine(part_of_bg, M, (rows, cols))
    
    y, x = icon.shape
    part_of_bg = img[top:top+y, left:left+x]
    part_of_bg = cv2.bitwise_and(part_of_bg.copy(), part_of_bg.copy(), mask=icon)
    result = part_of_bg + part_of_fg
    img[top:top+y, left:left+x] = result
    

    return img

cap = cv2.VideoCapture(0)
glasses = cv2.imread('eyeglasses.png')
detector = FaceMeshDetector(maxFaces=1)

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 1080))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, faces = detector.findFaceMesh(img, draw=False)
    
    if faces:
        face = faces[0]
        
        left_up = face[159]
        left_down = face[23]
        left_left = face[130]
        left_right = face[133]
        right_left = face[382]
        right_right = face[446]
        
        sx = left_left[0] - 50
        sy = left_up[1] - 50
        v = right_right[0] - left_left[0] + 100
        h = left_down[1] - left_up[1] + 100
    
        glasses1 = cv2.resize(glasses, (v, h))
        img = alphaMerge(img, glasses1, sy, sx)
        
        # cv2.putText(img, str(sx), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        # cv2.putText(img, str(sy), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        # cv2.putText(img, str(h), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        # cv2.putText(img, str(v), (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        
        # cv2.circle(img, left_up, 5, (255, 0, 255), 5)
        # cv2.circle(img, left_down, 5, (255, 0, 255), 5)
        # cv2.circle(img, left_left, 5, (255, 0, 255), 5)
        # cv2.circle(img, left_right, 5, (255, 0, 255), 5)
        # cv2.circle(img, right_left, 5, (255, 0, 255), 5)
        # cv2.circle(img, right_right, 5, (255, 0, 255), 5)
    

    cv2.imshow('img', img)
    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
