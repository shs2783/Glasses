import cv2
import numpy as np
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector


def alphaMerge(img, icon, top, left, angle=None):
    part_of_fg = icon.copy()
    fg_mask = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img_gray', fg_mask)

    fg_mask[fg_mask>0] = 255
    if fg_mask[:10,:10].sum() != 0:
        fg_mask = cv2.bitwise_not(fg_mask)
    part_of_fg = cv2.bitwise_and(part_of_fg, part_of_fg.copy(), fg_mask)
    
    if angle is not None:
        rows, cols = fg_mask.shape
        M = cv2.getRotationMatrix2D((int(rows/3.2), int(cols/3.2)), angle, scale=1)
        M[:, 2] += 50
        fg_mask = cv2.warpAffine(fg_mask, M, (int(cols*1.5), int(rows*2)))
        part_of_fg = cv2.warpAffine(part_of_fg, M, (int(cols*1.5), int(rows*2)))

    bg_mask = cv2.bitwise_not(fg_mask)
    
    y, x, _ = part_of_fg.shape
    top -= 50
    left -= 50
    part_of_bg = img[top:top+y, left:left+x]
    part_of_bg = cv2.bitwise_and(part_of_bg.copy(), part_of_bg.copy(), mask=bg_mask)
    result = part_of_bg + part_of_fg
    img[top:top+y, left:left+x] = result
        
    # cv2.imshow('fg_mask', fg_mask)
    # cv2.imshow('bg_mask', bg_mask)
    # cv2.imshow('part_of_fg', part_of_fg)
    # cv2.imshow('part_of_bg', part_of_bg)
    
    return img



def main():
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
            
            sx = left_left[0] - 40
            sy = left_up[1] - 100
            v = right_right[0] - left_left[0] + 80
            h = left_down[1] - left_up[1] + 200
            
            x = right_right[0] - left_left[0]
            y = right_right[1] - left_left[1]
            radians = np.arctan2(y, x)
            angle = -(radians * 180 / np.pi)

            glasses1 = cv2.resize(glasses, (v, h))
            img = alphaMerge(img, glasses1, sy, sx, angle)
        

        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:
            break
        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    glasses = cv2.imread('eyeglasses2.png')
    detector = FaceMeshDetector(maxFaces=1)
    main()