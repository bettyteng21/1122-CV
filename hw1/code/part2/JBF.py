import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        output = np.zeros_like(img)

        # spatial kernel LUT
        # here just first focus on diff of x (or y) only, the diff value will lie in 0~kernel_size
        # create LUT, where LUT index is the position diff ofa single axis
        Gs_LUT = np.exp((-1)*(np.arange(self.pad_w+1)**2)/(2*(self.sigma_s**2)))

        # range kernel LUT
        # diff of two pixel itensity always lies in (0~255)/255
        # create LUT, where LUT index is the intensity diff
        Gr_LUT = np.exp((-1)*((np.arange(256)/255)**2)/(2*(self.sigma_r**2)))

        upper = np.zeros_like(padded_img).astype(np.float64)
        lower = np.zeros_like(padded_img).astype(np.float64)

        # for gray guidance img 
        if guidance.ndim==2:
            # we are now standing in the another view
            # instead of looping through pixel-by-pixel, we only loop through the kernel size and focus on curr displacement
            for y in range(-self.pad_w, self.pad_w+1):
                for x in range(-self.pad_w, self.pad_w+1):

                    # calculate Gs
                    Gs = Gs_LUT[abs(x)]*Gs_LUT[abs(y)]

                    # we roll the img to simulate displacement, then calculate Gr
                    # displacement in axis 0 is y, axis 1 is x
                    Gr = Gr_LUT[np.abs(np.roll(padded_guidance, shift=(y,x), axis=(0,1))-padded_guidance)]

                    sr = Gs * Gr
                    for ch in range(3):
                        upper[:,:,ch] += sr*np.roll(padded_img[:,:,ch], shift=(y,x), axis=(0,1))
                        lower[:,:,ch] += sr

            for ch in range(3):
                # crop the padded boarder and calculate output
                output[:,:,ch] = upper[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w, ch]/lower[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w, ch]

        # for rgb guidance img, same procedure
        else:
            for y in range(-self.pad_w, self.pad_w+1):
                for x in range(-self.pad_w, self.pad_w+1):
                    Gs = Gs_LUT[abs(x)]*Gs_LUT[abs(y)]
                    
                    Gr = 1
                    for ch in range(3):
                        Gr *= Gr_LUT[np.abs(np.roll(padded_guidance[:,:,ch], shift=(y,x), axis=(0,1))-padded_guidance[:,:,ch])]
                    
                    sr = Gs * Gr
                    for ch in range(3):
                        upper[:,:,ch] += sr*np.roll(padded_img[:,:,ch], shift=(y,x), axis=(0,1))
                        lower[:,:,ch] += sr

            for ch in range(3):
                output[:,:,ch] = upper[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w, ch]/lower[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w, ch]
            
        
        return np.clip(output, 0, 255).astype(np.uint8)
