import numpy as np
 
def landmarks_to_features(hand_landmark):
    pts = np.array([[lm.x,lm.y,lm.z]
                    for lm in hand_landmark.landmark])
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts[:,:2],axis=1))
    if scale >0:
        pts = pts / scale
    return pts.flatten()