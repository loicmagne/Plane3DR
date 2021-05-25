import numpy as np
import jax.numpy as jnp
import open3d as o3d
import cv2
import math as m
import sys
from tqdm import tqdm

def imshow(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Rx(theta):
    return jnp.array([[ 1, 0            , 0            ],
                     [ 0, jnp.cos(theta),-jnp.sin(theta)],
                     [ 0, jnp.sin(theta), jnp.cos(theta)]])
  
def Ry(theta):
    return jnp.array([[ jnp.cos(theta), 0, jnp.sin(theta)],
                     [ 0            , 1, 0            ],
                     [-jnp.sin(theta), 0, jnp.cos(theta)]])
  
def Rz(theta):
    return jnp.array([[ jnp.cos(theta), -jnp.sin(theta), 0 ],
                     [ jnp.sin(theta), jnp.cos(theta) , 0 ],
                     [ 0            , 0             , 1 ]])

def rotationM(x,y,z):
    return jnp.dot(jnp.dot(Rx(x),Rx(y)),Rx(z))

def euler(M): 
    tol = sys.float_info.epsilon * 10
    if abs(M[0,0]) < tol and abs(M[1,0]) < tol:
        eul1 = 0
        eul2 = m.atan2(-M[2,0], M[0,0])
        eul3 = m.atan2(-M[1,2], M[1,1])
    else:   
        eul1 = m.atan2(M[1,0],M[0,0])
        sp = m.sin(eul1)
        cp = m.cos(eul1)
        eul2 = m.atan2(-M[2,0],cp*M[0,0]+sp*M[1,0])
        eul3 = m.atan2(sp*M[0,2]-cp*M[1,2],cp*M[1,1]-sp*M[0,1])
    return eul1, eul2, eul3  

def matrix_pose_to_euler(M):
    angles = euler(M)
    euler_pose = np.array([*angles,*M[:3,3]])
    return euler_pose

def euler_pose_to_matrix(e):
    M = np.eye(4)
    M[:-1,:-1] = rotationM(e[0],e[1],e[2])
    M[:-1,-1] = e[3:]
    return M

def bilinear_interpolate(im, pt):
    x,y = pt
    x0 = jnp.floor(x).astype(int)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(int)
    y1 = y0 + 1

    x0 = jnp.clip(x0, 0, im.shape[1]-1)
    x1 = jnp.clip(x1, 0, im.shape[1]-1)
    y0 = jnp.clip(y0, 0, im.shape[0]-1)
    y1 = jnp.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (Ia*wa) + (Ib*wb) + (Ic*wc) + (Id.T*wd)

def bilinear_interpolate_np(im, pt):
    x,y = pt
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (Ia*wa) + (Ib*wb) + (Ic*wc) + (Id.T*wd)


def inside(pt,w,h):
    return (pt[0]>=0) and (pt[1]>=0) and (pt[0]<w) and (pt[1]<h) 

def depth_from_plane_params(pts, segmentation, plane_parameters, intrinsics, intrinsic_inv = None):
    '''
    Return a depth map given a plane segmentation mask and the corresponding
    plane parameters

    Parameters
    ----------
    pts : np.ndarray (shape: (2,n))
        list of points where we want to compute depth
    segmentation : np.ndarray (shape: (n))
        segmentation[i] is the plane in which belong the i-th point
    plane_parameters : np.ndarray (shape: (3,k))
        plane_parameters[:,i] gives the parameters of the i-th plane
    intrisics : np.ndarray (shape: (3,3))
        instrinsic camera matrix
    mask : np.ndarray (shape: (n))
        optionnal, binary mask to mask out non planar regions
    '''
    K = intrinsics
    if intrinsic_inv is not None:
        K_inv = intrinsic_inv
    else:
        K_inv = np.linalg.inv(K)

    n = len(pts[0])
    homogeneous = np.vstack([pts,np.ones([1,n])])
    normalized_plane_pt = np.dot(K_inv,homogeneous)
    params = plane_parameters.T[segmentation].T
    dot_product = np.sum(params * normalized_plane_pt, axis=0)
    depths = 1./dot_product

    return depths



def img_gradient(img):
    '''
    Computes the gradient of an image

    Parameters
    ----------
    img : cv2.Mat
        image

    Returns
    -------
    grad : cv2.Mat
        gradient
    '''
    # First apply some gaussian fitler 
    gray_img = cv2.GaussianBlur(img,(5,5),0)
    
    # Compute gradients
    grad_x = cv2.Scharr(gray_img,cv2.CV_64F,1,0)
    grad_y = cv2.Scharr(gray_img,cv2.CV_64F,0,1)
    grad = cv2.magnitude(grad_x,grad_y)
    return grad

def highGradientPixels(frame,threshold=0.,mask=None):
    '''
    Computes the high gradient pixels of a frame and returns a binary mask
    corresponding to high gradient pixels of the original frame

    Parameters
    ----------
    frame : Frame
        Frame object
    treshold : int
        min value of high gradient pixels
    mask : np.array | None
        if not None, only the pixels within the mask will be kept

    Returns
    -------
    result : np.array
        binary map, result[x,y] == 1 iff the pixel (x,y) is a high
        gradient one
    pixels : np.array
        list of coordinates of high gradient pixels
    '''
    # Compute gradient
    grad = img_gradient(frame.raw.img_gray)

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


def uniformlySamplesPixels(frame,mask=None,n=1000):
    '''
    Sample random pixels from the image

    Parameters
    ----------
    frame : Frame
        Frame object
    n : int
        number of pixels to sample
    mask : np.array | None
        if not None, only the pixels within the mask will be kept

    Returns
    -------
    result : np.array
        binary map, result[x,y] == 1 iff the pixel (x,y) is a high
        gradient one
    pixels : np.array
        list of coordinates of high gradient pixels
    '''
    gray_img = frame.raw.img_gray
    new_mask = np.zeros_like(gray_img)

    if mask is None:
        mask = np.zeros_like(gray_img)+255

    candidates = np.nonzero(mask)
    candidates = np.vstack((candidates[1],candidates[0])).T

    n = min(n,len(candidates))
    selected = np.random.choice(len(candidates), n, replace=False)
    pixels = candidates[selected]
    new_mask[pixels[:,1],pixels[:,0]] = 255
    return new_mask, pixels

if __name__ == '__main__':
    from frame import Frame
    f = Frame('100')
    
    gradient = img_gradient(f.raw.img_gray)
    res_grad, p = highGradientPixels(f,mask=f.raw.planarMask)
    res_rand, p = uniformlySamplesPixels(f,mask=f.raw.planarMask,n=10000)

    cv2.imshow('true_grad',gradient/255.)
    cv2.imshow('grad',res_grad)
    cv2.imshow('rand',res_rand)
    cv2.imshow('truth',f.raw.img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()