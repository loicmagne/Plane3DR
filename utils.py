import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm

def imshow(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def highGradientPixels(frame,threshold=0.05):
    '''
    Computes the high gradient pixels of a frame and returns a binary mask
    corresponding to high gradient pixels of the original frame

    Parameters
    ----------
    frame : Frame
        Frame object
    
    treshold : int
        min value of high gradient pixels
    '''
    # First apply some gaussian fitler 
    blurred_img = cv2.GaussianBlur(frame.raw.img,(5,5),0)
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2GRAY)
    
    # Compute gradients
    grad_x = cv2.Scharr(gray_img,cv2.CV_64F,1,0)
    grad_y = cv2.Scharr(gray_img,cv2.CV_64F,0,1)
    grad = cv2.magnitude(grad_x,grad_y)

    # Apply NMS (non-maximum suppression)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    max_filter = cv2.dilate(grad, kernel)
    nms_result = cv2.compare(grad,max_filter,cv2.CMP_EQ)

    # Apply thresholding 
    grad = grad/np.max(grad)
    _,thresh_result = cv2.threshold(grad,threshold,255,cv2.THRESH_BINARY)
    
    # Combine both mask
    nms_result = np.uint8(nms_result)
    thresh_result = np.uint8(thresh_result)
    result = cv2.bitwise_and(nms_result,thresh_result)
    
    return result

if __name__ == '__main__':
    from frame import Frame
    f = Frame('100')
    res = highGradientPixels(f)
    imshow(res)