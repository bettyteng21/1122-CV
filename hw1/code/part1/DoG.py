import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        img_copy = image.copy()

        for level in range(self.num_octaves):
            gaussian_images_in_octave = []
            gaussian_images_in_octave.append(img_copy)

            for i in range(1, self.num_guassian_images_per_octave):
                img_gaussian = cv2.GaussianBlur(img_copy, ksize=(0, 0), sigmaX= self.sigma**i, sigmaY= self.sigma**i)
                gaussian_images_in_octave.append(img_gaussian)

            gaussian_images.append(gaussian_images_in_octave)
            img_last = gaussian_images_in_octave[-1]
            img_copy = cv2.resize(img_last, dsize=None, fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
            # print(len(gaussian_images_in_octave))
        

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []

        for level in range(self.num_octaves):
            DoG_images_in_octave = []

            for i in range(self.num_DoG_images_per_octave):
                img_dog = cv2.subtract(gaussian_images[level][i+1],gaussian_images[level][i])
                DoG_images_in_octave.append(img_dog)

                # path="./output/DoG"+str(level+1)+"-"+str(i+1)+".png"
                # cv2.imwrite(path, img_dog.astype(np.uint8))
            
            dog_images.append(DoG_images_in_octave)
            # print(len(DoG_images_in_octave))


        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
            
        keypoints = []

        for level in range(self.num_octaves):
            for i in range(1, self.num_DoG_images_per_octave-1):
                img_pre = dog_images[level][i-1]
                img_curr = dog_images[level][i]
                img_next = dog_images[level][i+1]

                for y in range(1, img_curr.shape[0]-1):
                    for x in range(1, img_curr.shape[1]-1):

                        # get the 26 node arround coordinate (x,y)
                        compare_list= np.concatenate((np.hstack(img_pre[y-1 : y+2, x-1 : x+2]), np.hstack(img_next[y-1 : y+2, x-1 : x+2])), axis=None)
                        tmp_curr= np.hstack(img_curr[y-1 : y+2, x-1 : x+2])
                        tmp_curr= tmp_curr[np.arange(len(tmp_curr))!=4]  # exclude pixel (x,y)
                        compare_list= np.concatenate((compare_list, tmp_curr), axis=None)

                        # find local extreme
                        if(img_curr[y, x]<=np.min(compare_list) or img_curr[y, x]>=np.max(compare_list)):
                            # thresholding
                            if(np.abs(img_curr[y, x]) <= self.threshold):
                                continue

                            keypoints.append([y*np.power(2, level), x*np.power(2, level)])
  

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique

        keypoints = np.unique(keypoints, axis=0)                
        

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 

        # print(keypoints)

        return keypoints
