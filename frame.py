import numpy as np
import open3d as o3d
import cv2

import sys, os
from tqdm import tqdm
import pickle as pickle

sys.path.insert(0, os.path.abspath('./PlanarReconstruction/'))
import PlanarReconstruction.infer as infer

from util import depth_from_plane_params

class PlaneNet():
    '''
    Static Class to run inference of PlaneNet
    '''
    net = infer.Predictor()

    @staticmethod
    def infer(img):
        '''
        Run inference

        Parameters
        ----------
        img : string
            path  without extension to the image on which to run inference
        '''
        depth,seg,mask,params = PlaneNet.net.predict(f'data/raw_images/{img}.jpg')
        return depth,seg,mask,params

class RawFrame():
    def __init__(self,img,depthTreshold=None,planar=True):
        '''
        Class which defines a Frame of a sequence, and keep all the related
        informations (depth map, segmentation, plane parameters, etc...), without
        Open3D attributes, so it can be pickled

        Parameters
        ----------
        img : string
            path to the image without extension
        
        depthTreshold : int
            maximum accepted depth, usefull to filter far depth which are
            often irrelevant

        planar : boolean
            if true, only the planar parts of the image will be kept, if
            false, a depth prediction will be done on non planar parts
        '''
        self.img_path = img
        self.img = cv2.imread(f'data/raw_images/{img}.jpg')
        self.depth, self.segmentation, self.planarMask, self.planeParams = \
            PlaneNet.infer(self.img_path)
        self.planarMask = (255*self.planarMask).astype(np.uint8)

        # self.planeParams /= np.square(np.linalg.norm(self.planeParams.T,axis=1).T)

        # Resize image to the size of the depth map
        self.h, self.w = 192, 256
        self.img = cv2.resize(self.img, (self.w, self.h))
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Remove non planar areas and far areas
        self.planar = planar
        self.depthThreshold = depthTreshold
        if planar:
            self.depth[self.planarMask==0] = 0
        if depthTreshold:
            self.depth[self.depth>=depthTreshold] = 0

        # Compute intrinsics
        intrinsic = self.o3d_intrinsics()
        self.intrinsic = intrinsic.intrinsic_matrix

        # Compute point cloud
        self.cloud = self.depth_to_cloud(self.depth)

    def o3d_intrinsics(self):
        f = 1170.
        cx, cy = 128, 96
        scale_x = self.w / 1296.
        scale_y = self.h / 968.
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(self.w,self.h,f*scale_x,f*scale_y,cx,cy)
        return intrinsic

    def depth_to_cloud(self,depth):
        intrinsic = self.o3d_intrinsics()
        depth_img = o3d.geometry.Image(depth)
        cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_img,intrinsic)
        return np.asarray(cloud.points)

    def save(self):
        with open(f'data/precomputed_frames/{self.img_path}.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(img):
        with open(f'data/precomputed_frames/{img}.pkl', 'rb') as f:
            return pickle.load(f)

class Frame():
    def __init__(self,img,depthTreshold=None,planar=True,precomputed=True):
        '''
        Class which defines a Frame of a sequence, and keep all the related
        informations (depth map, segmentation, plane parameters, etc...), which will
        be used during computations

        Parameters
        ----------
        img : string
            path to the image without extension
        
        depthTreshold : int
            maximum accepted depth, usefull to filter far depth which are
            often irrelevant

        planar : boolean
            if true, only the planar parts of the image will be kept, if
            false, a depth prediction will be done on non planar parts
        '''
        if precomputed:
            self.raw = RawFrame.load(img)
        else:
            self.raw = RawFrame(img,depthTreshold,planar)

        self.cloud = o3d.geometry.PointCloud()
        self.cloud.points = o3d.utility.Vector3dVector(self.raw.cloud)
        self.cloudParam = o3d.geometry.PointCloud()
        self.cloudParam.points = o3d.utility.Vector3dVector(self.raw.planeParams.T)

    def updatePlaneParams(self, planeParams):
        self.raw.planeParams = planeParams

        planar_pts = np.nonzero(self.raw.planarMask)
        pts = np.vstack([planar_pts[1],planar_pts[0]])
        segmentation = self.raw.segmentation[planar_pts]

        new_depths = depth_from_plane_params(pts,segmentation-1,planeParams,self.raw.intrinsic)

        new_depth_map = np.zeros_like(self.raw.depth)
        new_depth_map[planar_pts] = new_depths
        new_depth_map[new_depth_map>=self.raw.depthThreshold] = 0

        self.raw.depth = new_depth_map
        self.raw.cloud = self.raw.depth_to_cloud(self.raw.depth)

        self.cloudParam.points = o3d.utility.Vector3dVector(self.raw.planeParams.T)
        self.cloud.points = o3d.utility.Vector3dVector(self.raw.cloud)

    def save(self):
        self.raw.save()