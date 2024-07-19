import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

def read_setting_file(setting_path):
    rgb_weight_list = []
    with open(setting_path, 'r') as f:
        next(f)
        lines = f.readlines()
        for line in lines[:-1]:
            s = line.replace('\n','').split(',')
            tmp = []
            for i in range(3):
                tmp.append(float(s[i]))
            rgb_weight_list.append(tmp)

        s = lines[-1].split(',')
        sigma_s = int(s[1])
        sigma_r = float(s[3])

        # print(rgb_weight_list)
        # print(sigma_s)
        # print(sigma_r)
        return rgb_weight_list, sigma_s, sigma_r

def rgb_to_grayscale(img, rgb_weight_list):
    gray_img_list = []

    for weight in rgb_weight_list:
        img_gray = weight[0]*img[:,:,0] + weight[1]*img[:,:,1] + weight[2]*img[:,:,2]
        gray_img_list.append(img_gray.astype(np.uint8))

    return gray_img_list




def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    # Step 1: read setting file
    rgb_weight_list = []
    rgb_weight_list, sigma_s, sigma_r= read_setting_file(args.setting_path)
    
    # Step 2: change to gray scale image
    gray_img_list = []
    gray_img_list = rgb_to_grayscale(img_rgb, rgb_weight_list)
    gray_img_list.append(img_gray)

    file_name=0
    for x in gray_img_list:
        file_name = file_name+1
        path="./output/gray/"+str(file_name)+".png"
        cv2.imwrite(path, x)

    # Step 3: Joint bilateral filter (6 gray img as guide)
    jbf_img_list = []
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    for gray in gray_img_list:
        img_processed = JBF.joint_bilateral_filter(img_rgb, gray) 
        jbf_img_list.append(img_processed)

    file_name=0
    for x in jbf_img_list:
        file_name = file_name+1
        path="./output/jbf/"+str(file_name)+".png"
        cv2.imwrite(path, cv2.cvtColor(x,cv2.COLOR_RGB2BGR))

    # Step 4: bilateral filter on origin rgb img
    img_bf = JBF.joint_bilateral_filter(img_rgb, img_rgb)

    # Step 5: calculate cost
    cost_list = []
    for jbf in jbf_img_list:
        cost = np.sum(np.abs(jbf.astype(np.int32) - img_bf.astype(np.int32)))
        cost_list.append(cost)
    
    print(cost_list)




if __name__ == '__main__':
    main()