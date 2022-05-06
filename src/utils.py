import numpy as np

#####開心開心####
def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = []  
    for r in range(N): 
        #print(m[r, 0])
        A.append([-u[r,0], -u[r,1], -1, 0, 0, 0, u[r,0]*v[r,0], u[r,1]*v[r,0], v[r,0]])
        A.append([0, 0, 0, -u[r,0], -u[r,1], -1, u[r,0]*v[r,1], u[r,1]*v[r,1], v[r,1]])

    # TODO: 2.solve H with A
    _, _, vt = np.linalg.svd(A, full_matrices=True) # Solve s ystem of linear equations Ah = 0 using SVD
    # pick H from last line of vt  
    H = np.reshape(vt[8], (3,3))
    # normalization, let H[2,2] equals to 1
    H = (1/H.item(8)) * H

    return H

def bilinear_interpolation(x, y, img):
    x1 = np.floor(x).astype(int)
    x2 = x1 + 1
    x1 = np.clip(x1, 0, img.shape[0]-1) # [0, im.shape[0]-1]
    x2 = np.clip(x2, 0, img.shape[0]-1)
    y1 = np.floor(y).astype(int)
    y2 = y1 + 1
    y1 = np.clip(y1, 0, img.shape[1]-1)
    y2 = np.clip(y2, 0, img.shape[1]-1)
    Ia = img[ x1, y1 ]
    Ib = img[ x2, y1 ]
    Ic = img[ x1, y2 ]
    Id = img[ x2, y2 ]
    wa = ((x2-x) * (y2-y)).reshape(-1,1)
    wb = ((x-x1) * (y2-y)).reshape(-1,1)
    wc = ((x2-x) * (y-y1)).reshape(-1,1)
    wd = ((x-x1) * (y-y1)).reshape(-1,1)
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, _ = src.shape
    h_dst, w_dst, _ = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    xs_range, ys_range = np.arange(w_src), np.arange(h_src)
    xd_range, yd_range = np.arange(w_dst), np.arange(h_dst)
    x_src, y_src = np.meshgrid(xs_range, ys_range)
    x_dst, y_dst = np.meshgrid(xd_range, yd_range)
    pixel = np.vstack([x_src.ravel(), y_src.ravel()]).T
    pixel_src = np.c_[pixel, np.ones(pixel.shape[0])]
    pixel = np.vstack([x_dst.ravel(), y_dst.ravel()]).T
    pixel_dst = np.c_[pixel, np.ones(pixel.shape[0])]

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        pixel_dst_t = pixel_dst.T
        homo_dst = np.dot(H_inv, pixel_dst_t)
        homo_image = np.vstack((homo_dst[1]/homo_dst[2],homo_dst[0]/homo_dst[2], pixel_dst_t[0], pixel_dst_t[1]))
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        height_filtered1 = homo_image[:, np.ceil(homo_image[0, :]) < h_src ]
        height_filtered2 = height_filtered1[:, np.floor(height_filtered1[0, :]) > 0 ]
        masked_src1 = height_filtered2[:, np.ceil(height_filtered2[1, :]) < w_src]
        masked_src = (masked_src1[:, np.floor(masked_src1[1, :]) > 0]).T
        new_img = bilinear_interpolation(masked_src[:, 0], masked_src[:, 1], src)
        # TODO: 6. assign to destination image with proper masking
        dst[np.floor(masked_src[: ,3]).astype(int), np.floor(masked_src[: ,2]).astype(int)] = new_img


    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        pixel_src_t = pixel_src.T
        homo_src = np.dot(H, pixel_src_t)
        i_src = src.reshape(-1,3).T
        homo_image = np.vstack((homo_src[1]/homo_src[2],homo_src[0]/homo_src[2], i_src))
        
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        height_filtered1 = homo_image[:, homo_image[0, :] < h_dst ]
        height_filtered2 = height_filtered1[:, height_filtered1[0, :] > 0 ]
        masked_src1 = height_filtered2[:, height_filtered2[1, :] < w_dst]
        masked_src = (masked_src1[:, masked_src1[1, :] > 0].astype(int)).T
        
        # TODO: 6. assign to destination image using advanced array indicing
        dst[masked_src[: ,0], masked_src[: ,1]] = masked_src[:, 2:5]

    return dst
