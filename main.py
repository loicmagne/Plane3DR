import numpy as np
import open3d as o3d
from tqdm import tqdm

from frameseq import *

def point2pointICP(frameSeq,threshold=0.5,iterations=100):
    transf = [np.eye(4)]
    current = frameSeq[0]
    for nxt in tqdm(frameSeq[1:]):
        cloud1 = current.cloud
        cloud2 = nxt.cloud
        reg_p2p = o3d.pipelines.registration.registration_icp(
            cloud2, cloud1, threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        )
        current = nxt
        transf.append(reg_p2p.transformation)
    return transf

def plane2planeICP(frameSeq,threshold=0.5,iterations=100):
    transf = [np.eye(4)]
    current = frameSeq[0]
    for nxt in tqdm(frameSeq[1:]):
        cloud1 = current.cloudParam
        cloud2 = nxt.cloudParam
        reg_p2p = o3d.pipelines.registration.registration_icp(
            cloud2, cloud1, threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        )
        current = nxt
        transf.append(reg_p2p.transformation)
    return transf

if __name__ == '__main__':
    seq = FrameSeq([str(10*k) for k in range(1)],precomputed=True)
    seq.display()
    seq.transform(point2pointICP(seq))
    print(np.asarray(seq[0].cloudParam.points))
    seq.display()