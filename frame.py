import numpy as np
import sys, os
import open3d as o3d
from tqdm import tqdm
import pickle as pickle

sys.path.insert(0, os.path.abspath('./PlanarReconstruction/'))
import PlanarReconstruction.infer as infer

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
        self.img = img
        self.depth, self.segmentation, self.planarMask, self.planeParams = \
            PlaneNet.infer(self.img)

        self.planeParams /= np.square(np.linalg.norm(self.planeParams.T,axis=1).T)
        if planar:
            self.depth[self.planarMask==0] = 0
        if depthTreshold:
            self.depth[self.depth>=depthTreshold] = 0

        f = 1170.
        w, h = 256, 192
        cx, cy = 128, 96
        scale_x = w / 1296.
        scale_y = h / 968.
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(w,h,f*scale_x,f*scale_y,cx,cy)
        depth_img = o3d.geometry.Image(self.depth)
        cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_img,intrinsic)
        self.cloud = np.asarray(cloud.points)

    def save(self):
        with open(f'data/precomputed_frames/{self.img}.pkl', 'wb') as f:
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
        self.cloudParam.points = o3d.utility.Vector3dVector(self.raw.planeParams.T)

    def save(self):
        self.raw.save()