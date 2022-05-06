import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

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
    for idx in range(len(imgs) - 1):
        threshold_inliers = 0
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        match_src =  np.array([keypoints1[matches[i].queryIdx].pt for i in range(len(matches))])
        match_dst =  np.array([keypoints2[matches[i].trainIdx].pt for i in range(len(matches))])
        # TODO: 2. apply RANSAC to choose best H
        best_H = None
        for i in range(5000):
            indices = random.sample(range(len(matches)), 4)
            random_src = np.array([match_src[index] for index in indices])
            random_dst = np.array([match_dst[index] for index in indices])
            H = solve_homography(random_dst, random_src)
            src_pts = np.hstack((match_src, np.ones((len(matches),1)))).T
            dst_pts = np.hstack((match_dst, np.ones((len(matches),1)))).T
            projected_pts = np.dot(H, dst_pts)
            projection_error = np.sqrt(np.sum(np.square(src_pts[:-1,:] - (projected_pts/projected_pts[-1])[:-1,:]), axis=0)).reshape(1,-1)
            src_pts = np.vstack((src_pts, projection_error))
            dst_pts = np.vstack((dst_pts, projection_error))
            if np.count_nonzero(projection_error < 12) > threshold_inliers:
                best_H = H
                threshold_inliers = np.count_nonzero(projection_error < 12)
        # TODO: 3. chain the homographies
        last_best_H_non = np.dot(last_best_H, best_H)
        last_best_H = last_best_H_non/last_best_H_non[-1,-1]
        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')
    out = dst
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)