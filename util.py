import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm

def imshow(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inside(pt,w,h):
    return (pt[0]>=0) and (pt[1]>=0) and (pt[0]<w) and (pt[1]<h) 

def highGradientPixels(frame,threshold=0.05,mask=None):
    '''
    Computes the high gradient pixels of a frame and returns a binary mask
    corresponding to high gradient pixels of the original frame

    Parameters
    ----------
    frame : Frame
        Frame object
    
    treshold : int
        min value of high gradient pixels

    Returns
    -------
    result : np.array
        binary map, result[x,y] == 1 iff the pixel (x,y) is a high
        gradient one
    pixels : np.array
        list of coordinates of high gradient pixels
    mask : np.array | None
        if not None, only the pixels within the mask will be kept
    '''
    # First apply some gaussian fitler 
    gray_img = cv2.GaussianBlur(frame.raw.img_gray,(5,5),0)
    
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

    # bitwise AND with mask
    if mask is not None:
        result = cv2.bitwise_and(result,mask)

    # Get high gradient pixels coordinates
    pixels = np.nonzero(result)
    pixels = np.vstack((pixels[1],pixels[0])).T

    return result, pixels

if __name__ == '__main__':
    from frame import Frame
    f = Frame('100')
    res, p = highGradientPixels(f,mask=f.raw.planarMask)
    imshow(res)
    print(p.shape)
    res, p = highGradientPixels(f)
    imshow(res)
    print(p.shape)