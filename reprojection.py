import jax.numpy as jnp
import numpy as np
from jax import jit, grad, jacfwd, jacrev
from numpy.matrixlib.defmatrix import matrix

from util import highGradientPixels

# PyCeres import
import sys
import os
import math as m
pyceres_location='../ceres-bin/lib'
sys.path.insert(0, os.path.abspath(pyceres_location))
import PyCeres

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



def costF(ref, tar, pts):
    K = ref.raw.intrinsic
    K_inv = jnp.linalg.inv(K)
    n = pts.shape[0]
    def cost(camera):
        rotation = rotationM(*camera[:3])
        translation = camera[3:].reshape(-1,1)
        # Compute the photometric residual
        homogeneous = jnp.concatenate((pts.T,jnp.ones((1,n))))
        normalized_plane_pt = jnp.dot(K_inv,homogeneous)
        depths = ref.raw.depth[pts.T[1].astype(int),pts.T[0].astype(int)]
        pt_3d =  normalized_plane_pt * depths
        projected_pt = jnp.dot(rotation,pt_3d) + translation
        tar_pt = jnp.dot(K,projected_pt)
        tar_pt = tar_pt[:2]/tar_pt[2]

        ref_i = bilinear_interpolate(ref.raw.img_gray/255., pts.T)
        tar_i = bilinear_interpolate(tar.raw.img_gray/255., tar_pt)
        
        res = ref_i - tar_i
        return jnp.sum(jnp.square(res))
    return cost


class PoseReprojectionCost(PyCeres.CostFunction):
    def __init__(self, ref_frame, tar_frame, pts):
        super().__init__()
        self.set_num_residuals(1) 
        self.set_parameter_block_sizes([6])

        self.ref = ref_frame
        self.tar = tar_frame
        self.pts = pts
        
        self.cost_function = costF(self.ref, self.tar, self.pts)
        self.cost_grad = grad(self.cost_function)

    # The CostFunction::Evaluate(...) virtual function implementation
    def Evaluate(self, parameters, residuals, jacobians):
        camera = parameters[0][:]
        residuals[0] = self.cost_function(camera)
        if (jacobians!=None):
            jacobians[0][:] = self.cost_grad(camera)

        return True
    


def costF_jac(ref, tar, pts):
    K = ref.raw.intrinsic
    K_inv = jnp.linalg.inv(K)
    n = pts.shape[0]
    def cost(camera):
        rotation = rotationM(*camera[:3])
        translation = camera[3:].reshape(-1,1)
        # Compute the photometric residual
        homogeneous = jnp.concatenate((pts.T,jnp.ones((1,n))))
        normalized_plane_pt = jnp.dot(K_inv,homogeneous)
        depths = ref.raw.depth[pts.T[1].astype(int),pts.T[0].astype(int)]
        pt_3d =  normalized_plane_pt * depths
        projected_pt = jnp.dot(rotation.T,pt_3d) - translation
        tar_pt = jnp.dot(K,projected_pt)
        tar_pt = tar_pt[:2]/tar_pt[2]

        ref_i = bilinear_interpolate(ref.raw.img_gray/255., pts.T)
        tar_i = bilinear_interpolate(tar.raw.img_gray/255., tar_pt)
        
        res = ref_i - tar_i
        return res
    return cost


class PoseReprojectionCost_jac(PyCeres.CostFunction):
    def __init__(self, ref_frame, tar_frame, pts):
        super().__init__()
        self.n = pts.shape[0]
        self.set_num_residuals(self.n) 
        self.set_parameter_block_sizes([6])

        self.ref = ref_frame
        self.tar = tar_frame
        self.pts = pts
        
        self.cost_function = costF_jac(self.ref, self.tar, self.pts)
        self.cost_grad = jacfwd(self.cost_function)

    # The CostFunction::Evaluate(...) virtual function implementation
    def Evaluate(self, parameters, residuals, jacobians):
        camera = parameters[0]
        residuals[:] = self.cost_function(camera)

        if (jacobians!=None):
            jacobians[0][:] = self.cost_grad(camera).ravel()

        return True


def reprojectionMinimizationPose(frame1, frame2, initial_p, max_step=500, verbose=False):
    problem = PyCeres.Problem()
    camera = np.ravel(initial_p)

    _, pixels = highGradientPixels(frame1,mask=frame1.raw.planarMask)

    # Add cost function
    cost_function = PoseReprojectionCost(frame1,frame2,pixels)
    problem.AddResidualBlock(cost_function,None,camera)

    # Setup the solver options as in normal ceres
    options=PyCeres.SolverOptions()
    options.linear_solver_type=PyCeres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout=verbose
    options.max_num_iterations=max_step

    # Summary
    summary=PyCeres.Summary()

    PyCeres.Solve(options,problem,summary)
    if verbose:
        print(summary.FullReport())
    return camera

def optimizePoses(frameSeq,init):
    ref = frameSeq[0]
    transf = [init[0]]
    for k in range(1,len(frameSeq)):
        tar = frameSeq[k]
        initial_pose = matrix_pose_to_euler(init[k])
        new_pose = reprojectionMinimizationPose(ref,tar,initial_pose)
        transf.append(euler_pose_to_matrix(new_pose))
    return transf





from main import ICP
from frameseq import FrameSeq

def main():
    seq = FrameSeq([str(10*k) for k in range(21)][:4],precomputed=True)
    init = ICP(seq,registration_technique='point2plane')
    seq.transform(init)
    seq.display()
    opt = optimizePoses(seq,init)
    seq.transform(opt)
    seq.display()



if __name__ == '__main__':
    PROFILE = False

    if PROFILE:
        import cProfile
        import pstats
        cProfile.run('main()','profiling')
        p = pstats.Stats('profiling')
        p.sort_stats('cumulative').print_stats(25)
        p.sort_stats('time').print_stats(25)
    else:
        main()