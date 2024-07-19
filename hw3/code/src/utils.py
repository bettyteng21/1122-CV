import numpy as np


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

    ############## method 1 (solve SVD) ##############
    # TODO: 1.forming A
    A = np.zeros((2*N, 9))
    b = np.zeros(2*N)

    for i in range(N):
        A[i*2] = [-u[i, 0], -u[i, 1], -1, 0, 0, 0, u[i, 0]*v[i, 0], u[i, 1]*v[i, 0], v[i, 0]]
        A[(i*2)+1] = [0, 0, 0, -u[i, 0], -u[i, 1], -1, u[i, 0]*v[i, 1], u[i, 1]*v[i, 1], v[i, 1]]

    U, S, Vh = np.linalg.svd(A)
    # TODO: 2.solve H with A
    H = Vh[-1].reshape(3, 3)

    ############## method 2 (solve linear system) ##################
    # TODO: 1.forming A
    # srcx, srcy, dstx, dsty = u[:,0], u[:,1], v[:,0], v[:,1]
    # A = np.zeros((9,9)) # Note: will be wrong if N!=4
    # b= np.zeros(9)
    # b[8] = 1

    # for i in range(N):
    #     A[i*2] = np.array([-srcx[i], -srcy[i], -1, 0, 0, 0, srcx[i]*dstx[i], srcy[i]*dstx[i], dstx[i]])
    #     A[(i*2)+1] = np.array([0, 0, 0, -srcx[i], -srcy[i], -1, srcx[i]*dsty[i], srcy[i]*dsty[i], dsty[i]])
    
    # A[8,8] = 1  # assuming h9=1
    # # TODO: 2.solve H with A 
    # H = np.reshape(np.linalg.solve(A, b), (3,3))

    return H

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

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x_coord, y_coord = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    ones_coord = np.ones_like(x_coord)
    in_coords = np.stack([x_coord.ravel(), y_coord.ravel(), ones_coord.ravel()], axis=-1).T

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        out_coords = H_inv @ in_coords
        u_coord = out_coords[0] / out_coords[2]
        v_coord = out_coords[1] / out_coords[2]
        u_coord = u_coord.reshape(ymax - ymin, xmax - xmin)
        v_coord = v_coord.reshape(ymax - ymin, xmax - xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        valid_mask = ((u_coord >= 0) & (u_coord < w_src-1) & (v_coord >= 0) & (v_coord < h_src-1))
        
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        u_valid, v_valid = u_coord[valid_mask], v_coord[valid_mask]

        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax, xmin:xmax][valid_mask] = bilinear(src, u_valid, v_valid)
        # dst[ymin:ymax, xmin:xmax,:][valid_mask] = src[v_valid.astype('int'), u_valid.astype('int'),:]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        out_coords = H @ in_coords
        u_coord = out_coords[0] / out_coords[2]
        v_coord = out_coords[1] / out_coords[2]
        u_coord = u_coord.reshape(ymax - ymin, xmax - xmin)
        v_coord = v_coord.reshape(ymax - ymin, xmax - xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        valid_mask = ((u_coord >= 0) & (u_coord < w_dst) & (v_coord >= 0) & (v_coord < h_dst))

        # TODO: 5.filter the valid coordinates using previous obtained mask
        # interpolation: get floor int
        u_valid = u_coord[valid_mask].astype(int)
        v_valid = v_coord[valid_mask].astype(int)

        # TODO: 6. assign to destination image using advanced array indicing       
        dst[v_valid, u_valid,:] = src[ymin:ymax, xmin:xmax][valid_mask]

    return dst 

def bilinear(img, x, y):
    x1, y1 = np.floor(x).astype('int'), np.floor(y).astype('int')
    x2, y2 = x1+1, y1+1

    wa = np.repeat((y2 - y) * (x2 - x), 3).reshape((-1, 3))
    wb = np.repeat((x2 - x) * (y - y1), 3).reshape((-1, 3))
    wd = np.repeat((x - x1) * (y2 - y), 3).reshape((-1, 3))
    wc = np.repeat((x - x1) * (y - y1), 3).reshape((-1, 3))

    return wa * img[y1, x1] + wb * img[y2, x1] + wc * img[y2, x2] + wd * img[y1, x2]
