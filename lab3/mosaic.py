from __future__ import division
import sys
sys.path.append('/usr/local/Cellar/opencv/2.4.11_1/lib/python2.7/site-packages/')
from scipy.spatial import distance
import numpy as np
import cv2

CAM_KX = 720.
CAM_KY = CAM_KX
CAM_CX = 320.
CAM_CY = 200.

# DO NOT MODIFY cam_params_to_mat
def cam_params_to_mat(cam_kx, cam_ky, cam_cx, cam_cy):
    """Returns camera matrix K (3x3 numpy.matrix) from the focus and camera center parameters.
    """
    K = np.reshape(np.mat([cam_kx, 0, cam_cx, 0, cam_ky, cam_cy, 0, 0, 1]), (3,3))
    return K

# DO NOT MODIFY descript_keypt_extract
def descript_keypt_extract(img):
    """Takes an image and converts it to a list of descriptors and corresponding keypoints.

    From http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html#orb

    Returns:
        des (np.ndarray): Nx32 array of N feature descriptors of length 32.
        kpts (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        img2 (np.array): Image which draws the locations of found keypoints.
    """

    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kpts = orb.detect(img,None)

    # compute the descriptors with ORB
    kpts, des = orb.compute(img, kpts)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kpts, color=(255,0,0), flags=0)

    kpts = [np.mat([kpt.pt[0], kpt.pt[1], 1.0]).T for kpt in kpts]

    return des, kpts, img2

# fill your in code here
def propose_pairs(descripts_a, keypts_a, descripts_b, keypts_b):
    """Given a set of descriptors and keypoints from two images, propose good keypoint pairs.

    Feature descriptors should encode local image geometry in a way that is invariant to
    small changes in perspective. They should be comparible using an L2 metric.

    For the top N matching descrpitors, select and return the top corresponding keypoint pairs.

    Returns:
        pair_pts_a (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        pair_pts_b (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
    """
    # code here
    pair_pts_a = []
    pair_pts_b = []
    distances = []
    zipped_a = zip(descripts_a, keypts_a)
    zipped_b = zip(descripts_b, keypts_b)
    distance_product = [(cv2.norm(a[0],b[0],cv2.NORM_HAMMING), a[1], b[1]) for a in zipped_a for b in zipped_b]
    sorted_list = sorted(distance_product, key=lambda x: x[0])
    N = min(100, len(sorted_list))
    for n in range(N):
        pair_pts_a.append(sorted_list[n][1])
        pair_pts_b.append(sorted_list[n][2])
        distances.append(sorted_list[n][0])
    return pair_pts_a, pair_pts_b



# fill your in code here
def homog_dlt(ptsa, ptsb):
    """From a list of point pairs, find the 3x3 homography between them.

    Find the homography H using the direct linear transform method. For correct
    correspondences, the points should all satisfy the equality
    w*ptb = H pta, where w > 0 is some multiplier. 

    Arguments:
        ptsa (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        ptsb (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 

    Returns:
        H (np.matrix of shape 3x3): Homography found using DLT.
    """
    # code here
    """
    [u v 1 0 0 0 -uu' -vu' -u'...
     0 0 0 u v 1 -uv' -vv' -v']
    """
    ptsa_np = np.array([[ptsa[i].item(0) for i in range(0, len(ptsa))], [ptsa[i].item(1) for i in range(0, len(ptsa))]])
    ptsb_np = np.array([[ptsb[i].item(0) for i in range(0, len(ptsb))], [ptsb[i].item(1) for i in range(0, len(ptsb))]])
    ptsa_mean = np.mean(ptsa_np, axis=1)
    ptsb_mean = np.mean(ptsb_np, axis=1)
    scale_a = 1/np.sum(np.std(np.transpose(ptsa_np) - ptsa_mean, axis = 0, ddof = 1))
    scale_b = 1/np.sum(np.std(np.transpose(ptsb_np) - ptsb_mean, axis = 0, ddof = 1))
    norm_a = np.array([[1, 0, -ptsa_mean[0]], [0, 1, -ptsa_mean[1]], [0,0,1/scale_a]])
    norm_a = np.matrix(scale_a * norm_a)
    norm_b = np.array([[1, 0, -ptsb_mean[0]], [0, 1, -ptsb_mean[1]], [0,0,1/scale_b]])
    norm_b = np.matrix(scale_b * norm_b)

    A = np.zeros((len(ptsa) * 2, 9))
    for i in xrange(0, len(ptsa)):
        pta = norm_a * ptsa[i]
        ptb = norm_b * ptsb[i]
        u = pta[0, 0] / pta[2, 0]
        v = pta[1, 0] / pta[2, 0]
        w = ptb[2, 0]
        u_ = ptb[0, 0] / w
        v_ = ptb[1, 0] / w
        A[2 * i, :] = [u, v, 1, 0, 0, 0, -u * u_, -v * u_, -u_]
        A[2 * i + 1, :] = [0, 0, 0, u, v, 1, -u * v_, -v * v_, -v_]
    U, S, VT = np.linalg.svd(A)
    V = VT.T
    h = V[:, -1]
    H = np.mat(h).reshape((3, 3))
    H = norm_b.I * H * norm_a
    H = H / H[-1, -1]
    return H


# fill your in code here
def homog_ransac(pair_pts_a, pair_pts_b):
    """From possibly good keypoint pairs, determine the best homography using RANSAC.

    For a set of possible pairs, many of which are incorrect, determine the homography
    which best represents the image transformation. For the best found homography H,
    determine which points are close enough to be considered inliers for this model
    and return those.

    Returns:
        H (np.matrix of shape 3x3): Homography found using DLT and RANSAC.
        best_inliers_a (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        best_inliers_b (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
    """
    # code here
    max_iteration = 10000
    threhold = 0.80
    best_ratio = 0
    best_inliers_a = None
    best_inliers_b = None
    best_H = None
    for i in xrange(0, max_iteration):
        # sample_a = random.sample(pair_pts_a, 4)
        # sample_b = random.sample(pair_pts_b, 4)
        sample_indx = np.random.choice(len(pair_pts_a), 4, replace=False).tolist()
        sample_a = [pair_pts_a[i] for i in sample_indx]
        sample_b = [pair_pts_b[i] for i in sample_indx]
        try:
            H = homog_dlt(sample_a, sample_b)
        except np.linalg.linalg.LinAlgError:
            continue
        inliers_a, inliers_b = calculate_inliers(pair_pts_a, pair_pts_b, H)
        ratio = len(inliers_a) / len(pair_pts_a)
        if ratio >= threhold:
            return H, inliers_a, inliers_b
        if ratio > best_ratio:
            best_ratio = ratio
            best_H = H
            best_inliers_a = inliers_a
            best_inliers_b = inliers_b
    print "best ratio is %2.2f"%(best_ratio)
    print "number of inliers are %2.2f"%(len(best_inliers_a))
    return best_H, best_inliers_a, best_inliers_b


def calculate_inliers(pair_pts_a, pair_pts_b, H):
    assert len(pair_pts_a) == len(pair_pts_b)
    inliers_a = []
    inliers_b = []
    for i in xrange(0, len(pair_pts_a)):
        pta = pair_pts_a[i]
        ptb = pair_pts_b[i]
        homo_ptb = H * pta
        homo_ptb = homo_ptb / homo_ptb[2, 0]
        dist = np.linalg.norm(homo_ptb - ptb)
        if dist <= 15:
            inliers_a.append(pta)
            inliers_b.append(ptb)
    return inliers_a, inliers_b

# DO NOT MODIFY perspect_combine
def perspect_combine(img_a, img_b, H, length, width):
    """Perspective warp and blend two images based on given homography H.

    Create img_ab by first warping all the pixels in img_a according to the relation
    img_b = H img_a.  For pixels in both img_a and img_b, average the intensity.
    Otherwise, pick the image value present, and black otherwise.
    """
    bw = img_b.shape[0]
    bl = img_b.shape[1]

    warp_a = cv2.warpPerspective(img_a, H, (length, width))
    mask_a = np.zeros((bw, bl, 3))
    for i in range(bw):
        for j in range(bl):
            if np.sum(warp_a[i][j]) > 0.:
                mask_a[i,j,:] = 1.
    img_ab = warp_a
    img_ab[:bw,:bl,:] -= warp_a[:bw,:bl,:]/2.*mask_a[:bw,:bl,:]
    img_ab[:bw,:bl,:] += img_b*(1-mask_a[:bw,:bl,:])
    img_ab[:bw,:bl,:] += img_b/2.*mask_a[:bw,:bl,:]
    return img_ab

# DO NOT MODIFY img_combine_homog
def img_combine_homog(img_a, img_b, length_ab, width_ab):
    """Perspective warp and blend two images of nearby perspectives.
    """
    descripts_a, keypts_a, img_keypts_a = descript_keypt_extract(img_a)
    descripts_b, keypts_b, img_keypts_b = descript_keypt_extract(img_b)

    if True:
        cv2.imshow('Keypts A', img_keypts_a)
        cv2.imshow('Keypts B', img_keypts_b)

    pair_pts_a, pair_pts_b = propose_pairs(descripts_a, keypts_a, descripts_b, keypts_b)
    assert(len(pair_pts_a) == len(pair_pts_b))
    assert(np.shape(pair_pts_a[0]) == (3,1) and np.shape(pair_pts_b[0]) == (3,1))
    assert(isinstance(pair_pts_a[0], np.matrix) and isinstance(pair_pts_b[0], np.matrix))

    best_H, best_inliers_a, best_inliers_b = homog_ransac(pair_pts_a, pair_pts_b)
    assert(np.shape(best_H) == (3,3) and isinstance(best_H, np.matrix))

    # cv2.drawKeypoints(img_a, best_inliers_a, (255, 0, 0), flags=0)
    # cv2.drawKeypoints(img_b, best_inliers_b, (255, 0, 0), flags=0)

    fixed_H = homog_dlt(best_inliers_a, best_inliers_b)
    #fixed_H = best_H
    assert(np.shape(fixed_H) == (3,3) and isinstance(fixed_H, np.matrix))

    img_ab = perspect_combine(img_a, img_b, fixed_H, length_ab, width_ab)
    return img_ab, fixed_H

# fill your in code here
def rot_from_homog(H, K):
    """Find the rotation matrix from a homography from perspectives with identical camera centers.
    
    The rotation found should be bRa or Ra^b.

    Arguments:
        H (np.matrix of shape 3x3): Homography
        K (np.matrix of shape 3x3): Camera matrix
    Returns:
        R (np.matrix of shape 3x3): Rotation matrix from frame a to frame b
    """
    # code here
    R = K.I * H * K
    return R


# fill your in code here
def extract_y_angle(R):
    """Given a rotation matrix around the y-axis, find the angle of rotation.

    The matrix need not be perfectly in SO(3), but provides an estimate nonetheless.

    Arguments:
        R (np.matrix of shape 3x3): Rotation matrix from frame a to frame b
    Returns:
        y_ang (float): angle in radians
    """
    # code here
    # y_ang = np.arctan(R.item(3) / R.item(0))
    return abs(np.arctan(-R.item(2) / R.item(8)))



# DO NOT MODIFY single_pair_combine
def single_pair_combine(img_ai, img_bi):
    img_a = cv2.imread('imageSet1/image_%02d.png'%img_ai)
    img_b = cv2.imread('imageSet1/image_%02d.png'%img_bi)

    # decimate by 2
    img_a = img_a[::2,::2,:]
    img_b = img_b[::2,::2,:]

    length_ab, width_ab = 1000, 600
    img_ab, best_H = img_combine_homog(img_a, img_b, length_ab, width_ab)

    K = cam_params_to_mat(CAM_KX, CAM_KY, CAM_CX, CAM_CY)
    R = rot_from_homog(best_H, K)
    assert(np.shape(R) == (3,3) and isinstance(R, np.matrix))
    y_ang = extract_y_angle(R)

    print 'H'
    print best_H
    print 'K'
    print K
    print 'R'
    print R
    print 'Y angle in Radians/Degrees'
    print y_ang, np.rad2deg(y_ang)

    cv2.imshow('Image A', img_a)
    cv2.imshow('Image B', img_b)
    cv2.imshow('Combined Image', img_ab)
    cv2.waitKey(0)

# DO NOT MODIFY multi_pair_combine
def multi_pair_combine(beg_i, n_imgs):
    #print 'ImageSet2/image2_%02d.png'%beg_i
    img_ab = cv2.imread('ImageSet2/image2_%02d.png'%(beg_i))
    img_ab = img_ab[::2,::2,:]
    for i in range(beg_i+1, beg_i+n_imgs):
        img_b = cv2.imread('ImageSet2/image2_%02d.png'%i)

        # decimate by 2
        img_b = img_b[::2,::2,:]

        length_ab, width_ab = 800+400*(i-1), 600
        img_ab, best_H = img_combine_homog(img_ab, img_b, length_ab, width_ab)

        cv2.imshow('Combined Image', img_ab)
        cv2.waitKey(100)
    cv2.imwrite('combo_%02d_%02d_set2.png'%(beg_i, beg_i+n_imgs-1), img_ab)


# DO NOT MODIFY main
def __main():
    if False:
        single_pair_combine(0, 1)

    if True:
        multi_pair_combine(1, 5)



if __name__ == "__main__":
    __main()