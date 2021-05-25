import numpy as np
import copy
import open3d as o3d
from tqdm import tqdm

from frame import *

def read_pose(path):
    '''
    Read the 3 pose as a 4x4 matrix from a file

    Parameters
    ----------
    path : string
        path the file

    Returns
    -------
    pose : np.array
        a 4x4 matrix corresponding to the pose
    '''
    return np.loadtxt(f'data/poses/{path}.txt')

class FrameSeq():
    '''
    Sequence of frames
    '''
    def __init__(self,imgs,depthTreshold=5,planar=True,precomputed=False,poses=False):
        '''
        Parameters
        ----------
        imgs : string list
            path to the image without extension
        
        depthTreshold : int
            maximum accepted depth, usefull to filter far depth which are
            often irrelevant

        planar : boolean
            if true, only the planar parts of the image will be kept, if
            false, a depth prediction will be done on non planar parts

        planar : boolean
            True to use precomputed features

        poses : boolean
            True to use real poses
        '''
        self.frames = []
        self.transformations = [] if poses else [np.eye(4) for k in range(len(imgs))]
        for img in tqdm(imgs):
            # Compute the frame and save it
            frame = Frame(img,depthTreshold,planar,precomputed)
            if not precomputed:
                frame.save()
            self.frames.append(frame)

            # Compute the pose and save it
            if poses:
                self.transformations.append(read_pose(img))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self,key):
        return self.frames[key]

    def transform(self,transformations,acc=True):
        '''
        Parameters
        ----------
        transformations : np.array list

        acc : boolean
            True if transformations are relative from frame to frame and
            should be accumulated
        '''
        if acc:
            for k in range(1,len(transformations)):
                transformations[k] = np.dot(
                                        transformations[k-1],
                                        transformations[k]
                                    )
        self.transformations = transformations

    def display(self):
        """
        # Rotates the point cloud to face the camera
        cloud.transform([[ 1 , 0 , 0 , 0 ], 
                         [ 0 ,-1 , 0 , 0 ],
                         [ 0 , 0 ,-1 , 0 ],
                         [ 0 , 0 , 0 , 1 ]])
        """

        cloud = []
        for f,t in zip(self.frames,self.transformations):
            copy_c = copy.deepcopy(f.cloud)
            copy_c.transform(t)
            cloud.append(copy_c)

            copy_c = copy.deepcopy(f.cloudParam)
            copy_c.transform(t)
            copy_c.paint_uniform_color([0,0,1])
            cloud.append(copy_c)

        o3d.visualization.draw_geometries(cloud)
    
    def displayPlaneParameters(self):
        transformed_clouds = []
        curr_t = self.transformations[0]
        for f,t in zip(self.frames,self.transformations):
            curr_t = np.dot(t,curr_t)
            copy_c = copy.deepcopy(f.cloudParam)
            copy_c.transform(curr_t)
            copy_c.paint_uniform_color(np.random.random(3))
            transformed_clouds.append(copy_c)
        o3d.visualization.draw_geometries(transformed_clouds)

if __name__ == '__main__':
    seq = FrameSeq([str(10*k) for k in range(20)],precomputed=True,poses=True)
    
    cloud = []

    f = seq[1]

    # Original
    copy_c = copy.deepcopy(f.cloud)
    copy_c.paint_uniform_color([0,0,1])
    cloud.append(copy_c)

    # Remapped params
    f.updatePlaneParams(f.raw.planeParams)
    copy_c = copy.deepcopy(f.cloud)
    copy_c.paint_uniform_color([1,0,0])
    cloud.append(copy_c)


    o3d.visualization.draw_geometries(cloud)