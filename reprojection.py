import jax.numpy as np
from jax import jit, grad, random

from util import highGradientPixels, inside

# PyCeres import
import sys
import os
pyceres_location='../ceres-bin/lib'
sys.path.insert(0, os.path.abspath(pyceres_location))
import PyCeres

def bilinear_interpolate(im, pt):
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

def costF(ref, tar, pt):
    K = ref.raw.intrinsic
    K_inv = np.linalg.inv(K)
    def cost(camera):
        # Compute the photometric residual
        homogeneous = np.concatenate((pt,np.array([1])))
        normalized_plane_pt = np.dot(K_inv,homogeneous)
        pt_3d =  normalized_plane_pt * ref.raw.depth[pt[1].astype(int),pt[0].astype(int)]
        pt_3d = np.concatenate((pt_3d,np.array([1.])))
        projected_pt = np.dot(camera,pt_3d)
        tar_pt = np.dot(K,projected_pt)
        tar_pt = tar_pt[:2]/tar_pt[2]

        ref_i = ref.raw.img_gray[pt[1].astype(int),pt[0].astype(int)]
        tar_i = bilinear_interpolate(tar.raw.img_gray, tar_pt)
        res = ref_i - tar_i
        return res
    return cost


class PoseReprojectionCost(PyCeres.CostFunction):
    def __init__(self, ref_frame, tar_frame, pt):
        super().__init__()
        self.set_num_residuals(1) 
        self.set_parameter_block_sizes([12])

        self.ref = ref_frame
        self.tar = tar_frame
        self.pt = pt
        
        self.cost_function = costF(self.ref, self.tar, self.pt)
        self.cost_grad = grad(self.cost_function)

    # The CostFunction::Evaluate(...) virtual function implementation
    def Evaluate(self, parameters, residuals, jacobians):
        camera = parameters[0].reshape((3,4))

        residuals[0] = self.cost_function(camera)

        if (jacobians!=None):
            jacobians[0][:] = self.cost_grad(camera).flatten()

        return True
    
def reprojectionMinimizationPose(frame1, frame2, initial_p):
    problem = PyCeres.Problem()
    camera = initial_p.flatten()
    _, pixels = highGradientPixels(frame1,mask=frame1.raw.planarMask)

    # Add each reprojection to the problem
    for pt in pixels:
        cost_function = PoseReprojectionCost(frame1,frame2,pt)
        problem.AddResidualBlock(cost_function,None,camera)

    # Setup the solver options as in normal ceres
    options=PyCeres.SolverOptions()
    options.linear_solver_type=PyCeres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout=True
    summary=PyCeres.Summary()
    PyCeres.Solve(options,problem,summary)
    print(summary.BriefReport() + " \n")
    print(camera)
    return camera

def optimizePoses(frameSeq,init):
    ref = frameSeq[0]
    transf = [np.eye(4)]
    for k in range(1,len(frameSeq)):
        tar = frameSeq[k]
        new_pose = reprojectionMinimizationPose(ref,tar,init[k][:3])
        new_pose = new_pose.reshape(3,4)
        new_pose = np.vstack((new_pose,[0.,0.,0.,1.]))
        transf.append(new_pose)
    return transf

if __name__ == '__main__':
    from main import *
    from frameseq import *
    from reprojection import *
    seq = FrameSeq([str(10*k) for k in range(21)][:2],precomputed=True)
    init = ICP(seq,registration_technique='point2plane')
    opt = optimizePoses(seq,init)
    seq.transform(opt)
    seq.display()