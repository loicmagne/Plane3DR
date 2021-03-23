import numpy as np
import open3d as o3d
from tqdm import tqdm

from frameseq import *

def ICP(frameSeq, registration_technique='point2point', inits=None, threshold=0.5, iterations=50):
    transf = [np.eye(4)]
    current = frameSeq[0]
    for k in tqdm(range(1,len(frameSeq))):
        nxt = frameSeq[k]
        init = np.eye(4) if (inits is None) else inits[k]
        reg_p2p = registrationICP(current, nxt, threshold, iterations, init,
                                  registration_technique)
        current = nxt
        transf.append(reg_p2p.transformation)
    return transf

def registrationICP(current, nxt, threshold, iterations, init, registration_technique):
    if registration_technique == 'point2point':
        cloud1 = current.cloud
        cloud2 = nxt.cloud
        reg_p2p = o3d.pipelines.registration.registration_icp(
            cloud2, cloud1, threshold, init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        )
    elif registration_technique == 'point2plane':
        cloud1 = current.cloud
        cloud2 = nxt.cloud
        cloud1.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        cloud2.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        reg_p2p = o3d.pipelines.registration.registration_icp(
            cloud2, cloud1, threshold, init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        )
    elif registration_technique == 'plane2plane':
        cloud1 = current.cloudParam
        cloud2 = nxt.cloudParam
        reg_p2p = o3d.pipelines.registration.registration_icp(
            cloud2, cloud1, threshold, init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        )
    return reg_p2p

if __name__ == '__main__':
    seq = FrameSeq([str(10*k) for k in range(21)][8:10],precomputed=True)
    seq.display()
    init = ICP(seq,registration_technique='point2point', inits=None)
    seq.transform(ICP(seq,registration_technique='point2plane', inits=init))
    seq.display()