import cv2
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev

from util import bilinear_interpolate_np, highGradientPixels, uniformlySamplesPixels, \
                 rotationM, matrix_pose_to_euler, bilinear_interpolate_np, \
                 euler_pose_to_matrix, bilinear_interpolate, \
                 img_gradient, imshow, depth_from_plane_params

# PyCeres import
import sys
import os
import math as m
pyceres_location='../ceres-bin/lib'
sys.path.insert(0, os.path.abspath(pyceres_location))
import PyCeres

def reproject(dest, src, transformation):
    '''
    Project the src frame into the dest frame given a transformation
    which maps points from src to dest
    '''
    K = dest.raw.intrinsic
    K_inv = np.linalg.inv(K)
    rotation = transformation[:-1,:-1]
    translation = transformation[:-1,-1].reshape(-1,1)
    h,w = dest.raw.h, dest.raw.w
    pts = np.nonzero(dest.raw.img_gray+1)
    pts = np.vstack((pts[1],pts[0]))

    homogeneous = np.vstack((pts,np.ones((1,h*w))))
    normalized_plane_pt = np.dot(K_inv,homogeneous)
    depths = dest.raw.depth[pts[1].astype(int),pts[0].astype(int)]
    pt_3d =  normalized_plane_pt * depths
    projected_pt = np.dot(rotation.T,pt_3d-translation)
    tar_pt = np.dot(K,projected_pt)
    tar_pt = tar_pt[:2]/tar_pt[2]

    reprojection = bilinear_interpolate_np(src.raw.img_gray, tar_pt)
    return reprojection.reshape(h,w)


# OPTIMIZATION OVER POSES ONLY

def costfPose(ref, tar, pts, grad_intensity=False):
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
        projected_pt = jnp.dot(rotation.T,pt_3d-translation)
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
        
        self.cost_function = costfPose(self.ref, self.tar, self.pts)
        self.cost_grad = jacfwd(self.cost_function)

    # The CostFunction::Evaluate(...) virtual function implementation
    def Evaluate(self, parameters, residuals, jacobians):
        camera = parameters[0]
        residuals[:] = self.cost_function(camera)

        if (jacobians!=None):
            jacobians[0][:] = self.cost_grad(camera).ravel()

        return True


# OPTIMIZATION OVER GEOMETRY (PLANES) ONLY

def costfGeometry(ref, tar, pts, transformation, grad_intensity=False):
    '''
    Parameters
    ----------
    grad_intensity : boolean
        use the intensity gradient instead of grayscale intensity
    '''
    K = ref.raw.intrinsic
    K_inv = jnp.linalg.inv(K)
    n = pts.shape[0]
    n_params = ref.raw.planeParams.shape[1]
    pts = pts.T

    ref_intensity = ref.raw.img_gray
    tar_intensity = tar.raw.img_gray
    if grad_intensity:
        ref_intensity = img_gradient(ref_intensity)
        tar_intensity = img_gradient(tar_intensity)

    segmentation = ref.raw.segmentation[pts[1],pts[0]]-1
    def cost(parameters):
        rotation = transformation[:-1,:-1]
        translation = transformation[:-1,-1].reshape(-1,1)
        plane_params = parameters.reshape(3,-1)
        # Compute the photometric residual
        homogeneous = jnp.concatenate((pts,jnp.ones((1,n))))
        normalized_plane_pt = jnp.dot(K_inv,homogeneous)
        depths = depth_from_plane_params(pts, segmentation, plane_params, K, K_inv)
        pt_3d =  normalized_plane_pt * depths
        projected_pt = jnp.dot(rotation.T,pt_3d-translation)
        tar_pt = jnp.dot(K,projected_pt)
        tar_pt = tar_pt[:2]/tar_pt[2]

        ref_i = bilinear_interpolate(ref_intensity, pts)
        tar_i = bilinear_interpolate(tar_intensity, tar_pt)
        
        res = ref_i - tar_i
        return res
    return cost


class GeometryReprojectionCost(PyCeres.CostFunction):
    def __init__(self, ref_frame, tar_frame, pts, transformation):
        super().__init__()
        self.n = pts.shape[0]
        self.n_params = ref_frame.raw.planeParams.shape[1]
        self.set_num_residuals(self.n) 
        self.set_parameter_block_sizes([3*self.n_params])

        self.ref = ref_frame
        self.tar = tar_frame
        self.pts = pts
        
        self.cost_function = costfGeometry(self.ref, self.tar, self.pts,transformation)
        self.cost_grad = jacfwd(self.cost_function)

    # The CostFunction::Evaluate(...) virtual function implementation
    def Evaluate(self, parameters, residuals, jacobians):   
        param = parameters[0]
        residuals[:] = self.cost_function(param)

        if (jacobians!=None):
            jacobians[0][:] = self.cost_grad(param).ravel()

        return True


# OPTIMIZATION OVER GEOMETRY + POSES

def costfPoseGeometry(ref, tar, pts, grad_intensity=False):
    '''
    Parameters
    ----------
    grad_intensity : boolean
        use the intensity gradient instead of grayscale intensity
    '''
    K = ref.raw.intrinsic
    K_inv = jnp.linalg.inv(K)
    n = pts.shape[0]
    n_params = ref.raw.planeParams.shape[1]
    pts = pts.T

    ref_intensity = ref.raw.img_gray
    tar_intensity = tar.raw.img_gray
    if grad_intensity:
        ref_intensity = img_gradient(ref_intensity)
        tar_intensity = img_gradient(tar_intensity)

    segmentation = ref.raw.segmentation[pts[1],pts[0]]-1
    def cost(parameters):
        rotation = rotationM(*parameters[:3])
        translation = parameters[3:6].reshape(-1,1)
        plane_params = parameters[6:].reshape(3,-1)
        # Compute the photometric residual
        homogeneous = jnp.concatenate((pts,jnp.ones((1,n))))
        normalized_plane_pt = jnp.dot(K_inv,homogeneous)
        depths = depth_from_plane_params(pts, segmentation, plane_params, K, K_inv)
        pt_3d =  normalized_plane_pt * depths
        projected_pt = jnp.dot(rotation.T,pt_3d-translation)
        tar_pt = jnp.dot(K,projected_pt)
        tar_pt = tar_pt[:2]/tar_pt[2]

        ref_i = bilinear_interpolate(ref_intensity, pts)
        tar_i = bilinear_interpolate(tar_intensity, tar_pt)
        
        res = ref_i - tar_i
        return res
    return cost


class PoseGeometryReprojectionCost(PyCeres.CostFunction):
    def __init__(self, ref_frame, tar_frame, pts):
        super().__init__()
        self.n = pts.shape[0]
        self.n_params = ref_frame.raw.planeParams.shape[1]
        self.set_num_residuals(self.n) 
        self.set_parameter_block_sizes([6+3*self.n_params])

        self.ref = ref_frame
        self.tar = tar_frame
        self.pts = pts
        
        self.cost_function = costfPoseGeometry(self.ref, self.tar, self.pts)
        self.cost_grad = jacfwd(self.cost_function)

    # The CostFunction::Evaluate(...) virtual function implementation
    def Evaluate(self, parameters, residuals, jacobians):
        param = parameters[0]
        residuals[:] = self.cost_function(param)

        if (jacobians!=None):
            jacobians[0][:] = self.cost_grad(param).ravel()

        return True


def reprojection_minimization(cost_function, init, verbose, max_step=500):
    variable = np.ravel(init)
    problem = PyCeres.Problem()
    # Add cost function
    loss_function = PyCeres.HuberLoss(0.05)
    cost_function = cost_function
    problem.AddResidualBlock(cost_function,loss_function,variable)

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
    return variable

def photometric_optimization(frameSeq,verbose=False,mode=2,pixel_sample='gradient'):
    '''
    mode : defines what should be optimized
        1 : only camera poses
        2 : only geometry
        3 : camera poses and geometry
    '''
    ref = frameSeq[0]
    poses = [np.eye(4)]
    for k in range(1,len(frameSeq)):
        tar = frameSeq[k]
        init_cp = None
        final_cp = None
        if pixel_sample == 'gradient':
            _, pixels = highGradientPixels(ref,mask=ref.raw.planarMask)
        elif pixel_sample == 'random':
            _, pixels = uniformlySamplesPixels(ref,mask=ref.raw.planarMask,n=5000)
        if mode == 1:
            initial_pose = matrix_pose_to_euler(frameSeq.rel_transformations[k])
            init_cp = np.copy(initial_pose)
            cost_function = PoseReprojectionCost(ref,tar,pixels)
            new_pose = reprojection_minimization(cost_function,initial_pose,verbose=verbose)
            final_cp = np.copy(new_pose)
            poses.append(euler_pose_to_matrix(new_pose))
        elif mode == 2:
            init = np.ravel(ref.raw.planeParams)
            init_cp = np.copy(init)
            cost_function = GeometryReprojectionCost(ref,tar,pixels,frameSeq.rel_transformations[k])
            new_params = reprojection_minimization(cost_function,init,verbose=verbose)
            final_cp = np.copy(new_params)
            ref.updatePlaneParams(new_params.reshape(3,-1))
        elif mode == 3:
            initial_pose = matrix_pose_to_euler(frameSeq.rel_transformations[k])
            initial_params = np.ravel(ref.raw.planeParams)
            init = np.concatenate([initial_pose,initial_params])
            init_cp = np.copy(init)
            cost_function = PoseGeometryReprojectionCost(ref,tar,pixels)
            new_pose = reprojection_minimization(cost_function,init,verbose=verbose)
            final_cp = np.copy(new_pose)
            poses.append(euler_pose_to_matrix(new_pose[:6]))
            ref.updatePlaneParams(new_pose[6:].reshape(3,-1))
        ref = tar
        if verbose:
            print('Initial parameters :')
            print(init_cp)
            print('Final parameters :')
            print(final_cp)
            print('Difference :')
            print(init_cp - final_cp)
    if mode == 1 or mode == 3:
        frameSeq.transform(poses)
    return poses

from main import ICP
from frameseq import FrameSeq

def main_ICP():
    seq = FrameSeq([0,10],precomputed=True)
    init = ICP(seq,registration_technique='point2plane')
    repro_init = reproject(seq[0],seq[1],init[1])
    seq.transform(init)
    seq.display()
    photometric_optimization(seq,True,mode=3,pixel_sample='gradient')
    seq.display()
    repro_opt = reproject(seq[0],seq[1],seq.rel_transformations[1])
    cv2.imshow("original",seq[0].raw.img_gray)
    cv2.imshow("reprojected init",repro_init/255.)
    cv2.imshow("reprojected opt",repro_opt/255.)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main_true():
    seq = FrameSeq([20,30],precomputed=True,poses=True)
    repro_init = reproject(seq[0],seq[1],seq.rel_transformations[1])
    seq.display()
    opt = photometric_optimization(seq,True,mode=2,pixel_sample='gradient')
    seq.display()
    repro_opt = reproject(seq[0],seq[1],seq.rel_transformations[1])
    cv2.imshow("original",seq[0].raw.img_gray)
    cv2.imshow("reprojected init",repro_init/255.)
    cv2.imshow("reprojected opt",repro_opt/255.)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main_ICP()