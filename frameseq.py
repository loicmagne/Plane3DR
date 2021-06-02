from jax.linear_util import transformation
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
        self.rel_transformations = [np.eye(4) for k in range(len(imgs))]
        for img in tqdm(imgs):
            # Compute the frame and save it
            frame = Frame(img,depthTreshold,planar,precomputed)
            if not precomputed:
                frame.save()
            self.frames.append(frame)

            # Compute the pose and save it
            if poses:
                self.transformations.append(read_pose(img))
        if poses:
            self.set_relative_transformations()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self,key):
        return self.frames[key]

    def set_relative_transformations(self):
        '''
        Update the self.rel_transformations attribute given the self.transformations
        is already set
        '''
        for k in range(1,len(self.transformations)):
            new_pose = np.eye(4)
            C_1, C_2 = self.transformations[k-1], self.transformations[k]
            R_1, R_2 = C_1[:-1,:-1], C_2[:-1,:-1]
            t_1, t_2 = C_1[:-1,-1], C_2[:-1,-1]
            new_rotation = np.dot(R_1.T,R_2)
            new_translation = np.dot(R_1.T,t_2-t_1)
            new_pose[:-1,:-1] = new_rotation
            new_pose[:-1,-1] = new_translation
            self.rel_transformations[k] = new_pose


    def set_absolute_transformations(self):
        '''
        Update the self.transformations attribute given the self.rel_transformations
        is already set
        '''
        for k in range(1,len(self.rel_transformations)):
            self.transformations[k] = np.dot(
                self.transformations[k-1],
                self.rel_transformations[k]
            )

    def transform(self,transformations,method='rel'):
        '''
        Parameters
        ----------
        transformations : np.array list

        mehthod : 'abs' or 'rel'
            'abs' for absolute poses, 'rel' for relative poses
        '''
        if method=='rel':
            self.rel_transformations = transformations
            self.set_absolute_transformations()
        if method=='abs':
            self.transformations = transformations
            self.set_relative_transformations()

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

        o3d.visualization.draw_geometries(cloud)

if __name__ == '__main__':
    seq = FrameSeq([str(10*k) for k in range(20)],precomputed=True,poses=True)
    seq.display()
    seq.set_absolute_transformations()
    seq.display()
