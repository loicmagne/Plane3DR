import numpy as np
import copy
import open3d as o3d
from tqdm import tqdm

from frame import *

class FrameSeq():
    '''
    Sequence of frames
    '''
    def __init__(self,imgs,depthTreshold=5,planar=True,precomputed=False):
        self.frames = []
        for img in tqdm(imgs):
            frame = Frame(img,depthTreshold,planar,precomputed)
            if not precomputed:
                frame.save()
            self.frames.append(frame)
        self.transformations = [np.eye(4) for k in range(len(self.frames))]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self,key):
        return self.frames[key]

    def transform(self,transformations):
        self.transformations = transformations

    def display(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]))
        pcd.colors = o3d.utility.Vector3dVector(np.array([[1,0,0],[0,1,0],[0,1,0],[0,1,0]]))

        """
        # Rotates the point cloud to face the camera
        cloud.transform([[ 1 , 0 , 0 , 0 ], 
                         [ 0 ,-1 , 0 , 0 ],
                         [ 0 , 0 ,-1 , 0 ],
                         [ 0 , 0 , 0 , 1 ]])"""

        transformed_clouds = [pcd]
        curr_t = self.transformations[0]
        for f,t in zip(self.frames,self.transformations):
            curr_t = np.dot(t,curr_t)
            copy_c = copy.deepcopy(f.cloud)
            copy_c.transform(curr_t)
            transformed_clouds.append(copy_c)

            copy_c = copy.deepcopy(f.cloudParam)
            copy_c.transform(curr_t)
            copy_c.paint_uniform_color([0,0,1])
            transformed_clouds.append(copy_c)

        o3d.visualization.draw_geometries(transformed_clouds)
    
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
    seq = FrameSeq([str(10*k) for k in range(5)])
    seq.display()