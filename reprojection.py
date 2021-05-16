import jax.numpy as jnp
import numpy as np
from jax import jit, grad, jacfwd, jacrev
from numpy.matrixlib.defmatrix import matrix

from util import highGradientPixels, uniformlySamplesPixels, \
                 rotationM, matrix_pose_to_euler, \
                 euler_pose_to_matrix, bilinear_interpolate, \
                 img_gradient

# PyCeres import
import sys
import os
import math as m
pyceres_location='../ceres-bin/lib'
sys.path.insert(0, os.path.abspath(pyceres_location))
import PyCeres

def costF(ref, tar, pts, grad_intensity=False):
    '''
    Parameters
    ----------
    grad_intensity : boolean
        use the intensity gradient instead of grayscale intensity
    '''
    K = ref.raw.intrinsic
    K_inv = jnp.linalg.inv(K)
    n = pts.shape[0]

    ref_intensity = ref.raw.img_gray
    tar_intensity = tar.raw.img_gray
    if grad_intensity:
        ref_intensity = img_gradient(ref_intensity)
        tar_intensity = img_gradient(tar_intensity)
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

        ref_i = bilinear_interpolate(ref_intensity, pts.T)
        tar_i = bilinear_interpolate(tar_intensity, tar_pt)
        
        res = ref_i - tar_i
        return res
    return cost


class PoseReprojectionCost(PyCeres.CostFunction):
    def __init__(self, ref_frame, tar_frame, pts):
        super().__init__()
        self.n = pts.shape[0]
        self.set_num_residuals(self.n) 
        self.set_parameter_block_sizes([6])

        self.ref = ref_frame
        self.tar = tar_frame
        self.pts = pts
        
        self.cost_function = costF(self.ref, self.tar, self.pts)
        self.cost_grad = jacfwd(self.cost_function)

    # The CostFunction::Evaluate(...) virtual function implementation
    def Evaluate(self, parameters, residuals, jacobians):
        camera = parameters[0]
        residuals[:] = self.cost_function(camera)

        if (jacobians!=None):
            jacobians[0][:] = self.cost_grad(camera).ravel()

        return True


def reprojectionMinimizationPose(frame1, frame2, initial_p, verbose, max_step=1000, pixel_sample='gradient'):
    problem = PyCeres.Problem()
    camera = np.ravel(initial_p)

    if pixel_sample == 'gradient':
        _, pixels = highGradientPixels(frame1,mask=frame1.raw.planarMask)
    elif pixel_sample == 'random':
        _, pixels = uniformlySamplesPixels(frame1,mask=frame1.raw.planarMask,n=5000)

    # Add cost function
    loss_function = PyCeres.HuberLoss(0.05)
    cost_function = PoseReprojectionCost(frame1,frame2,pixels)
    problem.AddResidualBlock(cost_function,loss_function,camera)

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

def optimizePoses(frameSeq,init,verbose=False,pixel_sample='gradient'):
    ref = frameSeq[0]
    transf = [init[0]]
    for k in range(1,len(frameSeq)):
        tar = frameSeq[k]
        initial_pose = matrix_pose_to_euler(init[k])
        new_pose = reprojectionMinimizationPose(ref,tar,initial_pose,verbose=verbose,pixel_sample=pixel_sample)
        transf.append(euler_pose_to_matrix(new_pose))
        ref = tar
    return transf


from main import ICP
from frameseq import FrameSeq

def main():
    seq = FrameSeq([100,110],precomputed=True)
    init = ICP(seq,registration_technique='point2plane')
    opt = optimizePoses(seq,init,True,pixel_sample='random')
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