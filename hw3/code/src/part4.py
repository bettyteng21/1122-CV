import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def RANSAC_choose_best_H(kp1, kp2, iter, sample_kp=8, threshold=0.5):
    rand_idx_range = min(len(kp1), len(kp2))
    best_H = np.eye(3)
    max_inlier = 0

    for i in range(iter):
        # TODO: 1.get an estimated H
        rand_idx = random.sample(range(rand_idx_range), sample_kp)
        rand_kp1, rand_kp2 = kp1[rand_idx], kp2[rand_idx]
        est_H = solve_homography(rand_kp2, rand_kp1)
        
        # TODO: 2.compare distance between estimated dst(H @ kp1) and actual dst (kp2)
        # arrange the form of kp1
        kp2_x, kp2_y = (np.stack(kp2, axis=1))[0], (np.stack(kp2, axis=1))[1]
        kp2_ones = np.ones_like(kp2_x)
        kp2_in = np.stack([kp2_x.ravel(), kp2_y.ravel(), kp2_ones.ravel()], axis=-1).T

        # get estimated kp2
        est_kp1_out = est_H @ kp2_in
        est_kp1_u, est_kp1_v = est_kp1_out[0]/est_kp1_out[2], est_kp1_out[1]/est_kp1_out[2]
        est_kp1 = np.stack((est_kp1_u.ravel(), est_kp1_v.ravel()), axis=1)

        # TODO: 3.caculate # of inlier points
        distance = np.linalg.norm((est_kp1 - kp1), axis=1)
        inlier = (distance <= threshold).sum()

        # TODO: 4.update best H
        if inlier > max_inlier :
            max_inlier = inlier
            # print(max_inlier)
            best_H = est_H.copy()
        
    return best_H

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
    # for idx in (range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()

        # get feature
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)

        # feature matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des2, des1)
        matches = sorted(matches, key = lambda x: x.distance)

        # get the index of matched descriptors
        t_idx = [match.trainIdx for match in matches]
        q_idx = [match.queryIdx for match in matches]

        # use the index to find corresponding keypoints
        kp1_list = np.array([kp1[idx].pt for idx in t_idx])
        kp2_list = np.array([kp2[idx].pt for idx in q_idx])

        # TODO: 2. apply RANSAC to choose best H
        H = RANSAC_choose_best_H(kp1_list, kp2_list, sample_kp=13, iter=9000, threshold=1)

        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ H

        # TODO: 4. apply stitching
        out = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)

