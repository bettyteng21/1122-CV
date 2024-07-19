import numpy as np
import cv2.ximgproc as xip # type: ignore
import cv2

def computeBinaryPattern(img, window_size=5):
    pad = window_size // 2
    padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')

    shift = [(i-pad, j-pad) for i in range(window_size) for j in range(window_size)]
    shift.remove((0,0)) # to not include the center point

    bp = np.zeros((img.shape[0], img.shape[1], len(shift)), dtype=bool)

    for s, (i,j) in enumerate(shift):
        # shift the padded_img to each direction in (i,j)
        shifted_img = padded_img[pad+i:pad+i+img.shape[0], pad+j:pad+j+img.shape[1]]

        # compare the origin img (center) with the shifted img (each direction)
        bp[:,:,s] = (img > shifted_img)

    return bp


def computeHammingDist(bp1, bp2):
    xor_bp = np.logical_xor(bp1, bp2)
    hamming_dist = np.sum(xor_bp,axis=2)
    return hamming_dist


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)

    Il_gray = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Ir_gray = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    Il_bp = computeBinaryPattern(Il_gray, window_size=11)
    Ir_bp = computeBinaryPattern(Ir_gray, window_size=11)


    # ppt p27 flow chart
    cost_volume_l2r = np.zeros((h, w, max_disp), dtype=np.float32)
    cost_volume_r2l = np.zeros((h, w, max_disp), dtype=np.float32)

    # cost volumes construction
    for d in range(max_disp):
        shifted_Ir_bp = np.pad(Ir_bp, ((0, 0), (d, 0), (0, 0)), mode='edge')[:, :w, :]
        shifted_Il_bp = np.pad(Il_bp, ((0, 0), (0, d), (0, 0)), mode='edge')[:, d:, :]

        cost_volume_l2r[:, :, d] = computeHammingDist(Il_bp, shifted_Ir_bp)
        cost_volume_r2l[:, :, d] = computeHammingDist(Ir_bp, shifted_Il_bp)


    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)

    for d in range(max_disp):
        cost_volume_l2r[:, :, d] = xip.jointBilateralFilter(Il_gray.astype(np.uint8), cost_volume_l2r[:, :, d].astype(np.uint8), 12, 4, 12)
        cost_volume_r2l[:, :, d] = xip.jointBilateralFilter(Ir_gray.astype(np.uint8), cost_volume_r2l[:, :, d].astype(np.uint8), 12, 4, 12)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all

    disparity_l2r = np.argmin(cost_volume_l2r, axis=2)
    disparity_r2l = np.argmin(cost_volume_r2l, axis=2)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering

    # Weighted median filtering
    disparity_l2r = xip.weightedMedianFilter(Il.astype(np.uint8), disparity_l2r.astype(np.uint8), 11)

    return disparity_l2r.astype(np.uint8)

