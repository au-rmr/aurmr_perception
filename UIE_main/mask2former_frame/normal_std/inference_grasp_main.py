import os
import cv2
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scipy.io as scio
from normal_std.policy import estimate_suction
import math
import matplotlib.pyplot as plt
import calendar
import time
sys.path.append("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/UIE_main/mask2former_frame")


split = 'test_seen'
camera = 'kinect'
save_root = '/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/UIE_main/result'


class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


class run_normal_std():
    def uniform_kernel(self, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel = kernel / kernel_size**2
        return kernel

    def grid_sample(self, pred_score_map, down_rate=20, topk=50):
        num_row = pred_score_map.shape[0] // down_rate
        num_col = pred_score_map.shape[1] // down_rate

        idx_list = []
        for i in range(num_row):
            for j in range(num_col):
                pred_score_grid = pred_score_map[i*down_rate:(
                    i+1)*down_rate, j*down_rate:(j+1)*down_rate]
                max_idx = np.argmax(pred_score_grid)

                max_idx = np.array([max_idx // down_rate, max_idx %
                                    down_rate]).astype(np.int32)

                max_idx[0] += i*down_rate
                max_idx[1] += j*down_rate
                idx_list.append(max_idx[np.newaxis, ...])

        idx = np.concatenate(idx_list, axis=0)
        suction_scores = pred_score_map[idx[:, 0], idx[:, 1]]
        sort_idx = np.argsort(suction_scores)
        sort_idx = sort_idx[::-1]

        sort_idx_topk = sort_idx[:topk]

        suction_scores_topk = suction_scores[sort_idx_topk]
        idx0_topk = idx[:, 0][sort_idx_topk]
        idx1_topk = idx[:, 1][sort_idx_topk]
        return suction_scores_topk, idx0_topk, idx1_topk

    def drawGaussian(self, img, pt, score, sigma=1):
        """Draw 2d gaussian on input image.
        Parameters
        ----------
        img: torch.Tensor
            A tensor with shape: `(3, H, W)`.
        pt: list or tuple
            A point: (x, y).
        sigma: int
            Sigma of gaussian distribution.
        Returns
        -------
        torch.Tensor
            A tensor with shape: `(3, H, W)`.
        """
        tmp_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
        tmpSize = 3 * sigma
        # Check that any part of the gaussian is in-bounds
        ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
        br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

        if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
            # If not, just return the image as is
            return img

        # Generate gaussian
        size = 2 * tmpSize + 1
        x = np.arange(0, size, 1, float)
        # print('x:', x.shape)l
        y = x[:, np.newaxis]
        # print('x:', x.shape)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) /
                   (2 * sigma ** 2)) * score
        g = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        tmp_img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g
        img += tmp_img

    def inference(self, rgb_img, depth_img, mask, grasp_point):
        count = 0
        # reading the depth mask and rgb file, also reading the meta file for intrinsic parameters
        # segmask_file = "/home/soofiyanatar/Documents/AmazonHUB/UIE-main/masks/label0.png"
        # depth_file = "/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/UIE_main/annotated_real_v1_resized/depth/depth_2.png"
        # rgb_file = "/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/UIE_main/annotated_real_v1_resized/images/scene_03/bin_1E/bin_1E_color_0006.png"

        # segmask_file = "/home/soofiyanatar/datasets/Full_Dataset/label_segnet.png"
        # depth_file = "/home/soofiyanatar/datasets/Full_Dataset/depth_segnet.png"
        # rgb_file = "/home/soofiyanatar/datasets/Full_Dataset/rgb_segnet.png"

        # Getting the depth image in meters and segmentation mask in its original state (grayscale)
        # depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

        depth = depth_img.astype(np.float32)
        print('depth shape', depth.shape)
        seg_mask = mask.astype(np.uint8)
        print("shape of rgb", rgb_img.shape)
        # applying contour to get the edges of the mask

        try:
            if(grasp_point.all() == None):
                contours = cv2.findContours(
                    seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                big_contour = max(contours, key=cv2.contourArea)

                # get rotated rectangle from contour
                # get its dimensions
                # get angle relative to horizontal from rotated rectangle
                rotrect = cv2.minAreaRect(big_contour)
                (center), (width, height), angle = rotrect
            else:
                center = grasp_point
        except:
            contours = cv2.findContours(
                seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            big_contour = max(contours, key=cv2.contourArea)

            # get rotated rectangle from contour
            # get its dimensions
            # get angle relative to horizontal from rotated rectangle
            rotrect = cv2.minAreaRect(big_contour)
            (center), (width, height), angle = rotrect

        # For RGB
        fx, fy = 903.14697265625, 903.5240478515625
        cx, cy = 640.839599609375, 355.6268310546875
        width = 1280
        height = 720

        # These values are for Image with dimension 360x160
        # cx, cy = 180.0646, 80.8015
        # width = 360
        # height = 160

        # These values are for Image with dimension 450x320
        # cx, cy = 225.08075, 161.603
        # width = 450
        # height = 320

        s = 1000.0
        camera_info = CameraInfo(width, height, fx, fy, cx, cy, s)

        heatmap, normals, point_cloud = estimate_suction(
            depth, seg_mask, camera_info)
        print(normals.shape)

        k_size = 15
        kernel = self.uniform_kernel(k_size)
        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
        heatmap = np.pad(heatmap, k_size//2)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        heatmap = F.conv2d(heatmap, kernel).squeeze().numpy()

        suction_directions = normals[int(center[1]), int(center[0]), :]
        print(suction_directions)

        # pridictions
        score_image = heatmap
        score_image *= 255

        score_image = score_image.clip(0, 255)
        score_image = score_image.astype(np.uint8)
        score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
        rgb_image = rgb_img
        rgb_image = 0.5 * rgb_image + 0.5 * score_image
        rgb_image = rgb_image.astype(np.uint8)

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/"+str(time_stamp)+"grasp1.png", rgb_image)

        # sampled suctions
        # origin is at top left side, so calculate as per the origin
        print("center ", center)
        # depth /= 1000.0
        center3D_depth_value = depth[int(center[1]), int(center[0])]

        pointz = center3D_depth_value
        pointx = (center[0] - cx) * pointz / fx
        pointy = (center[1] - cy) * pointz / fy
        center_3D = (pointx, pointy, pointz)

        print("center ", center_3D)
        score_image = np.zeros_like(heatmap)
        best_distance = sys.maxsize
        best_point = [pointx, pointy, pointz]

        self.drawGaussian(score_image, [int(center[0]), int(center[1])], 1.0, 3)
        euler_angle = [math.asin(suction_directions[0]), math.asin(
            suction_directions[1])]
        # print(center_3D)

        # print("Rotation Matrix ", )
        # print("directions", suction_directions[suction_number])
        print("direction angles x, y ", math.asin(
            suction_directions[0])*180/math.pi, math.asin(
            suction_directions[1])*180/math.pi)
        # print("translations", suction_translations[suction_number])

        score_image *= 255
        score_image = score_image.clip(0, 255)
        score_image = score_image.astype(np.uint8)
        score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
        rgb_image = rgb_img
        rgb_image = 0.5 * rgb_image + 0.5 * score_image
        rgb_image = rgb_image.astype(np.uint8)
        # cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/grasp2.png", rgb_image)
        cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/"+str(time_stamp)+"grasp2.png", rgb_image)
        # im = Image.fromarray(rgb_image)

        # visu_dir = save_root
        # os.makedirs(visu_dir, exist_ok=True)
        # im.save(visu_dir+'/%04d_sampled' % count+'.png')
        # count += 1
        # cv2.destroyAllWindows()
        # best_point_return = [best_point[0], best_point[1], 1.081]
        best_point_return = [float(best_point[0]), float(best_point[1]), float(best_point[2])]
        return best_point_return, euler_angle, center

    def inference_vlm(self, rgb_img, depth_img, mask, grasp_point):
        count = 0
        # reading the depth mask and rgb file, also reading the meta file for intrinsic parameters
        # segmask_file = "/home/soofiyanatar/Documents/AmazonHUB/UIE-main/masks/label0.png"
        # depth_file = "/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/UIE_main/annotated_real_v1_resized/depth/depth_2.png"
        # rgb_file = "/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/UIE_main/annotated_real_v1_resized/images/scene_03/bin_1E/bin_1E_color_0006.png"

        # segmask_file = "/home/soofiyanatar/datasets/Full_Dataset/label_segnet.png"
        # depth_file = "/home/soofiyanatar/datasets/Full_Dataset/depth_segnet.png"
        # rgb_file = "/home/soofiyanatar/datasets/Full_Dataset/rgb_segnet.png"

        # Getting the depth image in meters and segmentation mask in its original state (grayscale)
        # depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/"+str(time_stamp)+"depth.png", depth_img)
        cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/"+str(time_stamp)+"mask_first.png", mask)
        cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/"+str(time_stamp)+"rgb_first.png", rgb_img)


        depth = depth_img.astype(np.float32)
        print('depth shape', depth.shape)
        seg_mask = mask.astype(np.uint8)
        print("shape of rgb", rgb_img.shape)
        # applying contour to get the edges of the mask

        try:
            if(grasp_point.all() == None):
                contours = cv2.findContours(
                    seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                big_contour = max(contours, key=cv2.contourArea)

                # get rotated rectangle from contour
                # get its dimensions
                # get angle relative to horizontal from rotated rectangle
                rotrect = cv2.minAreaRect(big_contour)
                (center), (width, height), angle = rotrect
            else:
                center = grasp_point
        except:
            contours = cv2.findContours(
                seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            big_contour = max(contours, key=cv2.contourArea)

            # get rotated rectangle from contour
            # get its dimensions
            # get angle relative to horizontal from rotated rectangle
            rotrect = cv2.minAreaRect(big_contour)
            (center), (width, height), angle = rotrect

        # For RGB
        fx, fy = 903.14697265625, 903.5240478515625
        cx, cy = 640.839599609375, 355.6268310546875
        width = 1280
        height = 720

        s = 1000.0
        camera_info = CameraInfo(width, height, fx, fy, cx, cy, s)

        heatmap, normals, point_cloud = estimate_suction(
            depth, seg_mask, camera_info)
        print(normals.shape)

        k_size = 15
        kernel = self.uniform_kernel(k_size)
        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
        heatmap = np.pad(heatmap, k_size//2)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        heatmap = F.conv2d(heatmap, kernel).squeeze().numpy()

        # pridictions
        score_image = heatmap
        score_image *= 255

        score_image = score_image.clip(0, 255)
        score_image = score_image.astype(np.uint8)
        score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
        rgb_image = rgb_img
        rgb_image = 0.5 * rgb_image + 0.5 * score_image
        rgb_image = rgb_image.astype(np.uint8)

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/"+str(time_stamp)+"grasp1.png", rgb_image)

        ## Comment this after implementing the VLM function to extract the grasp point
        input_grasp_point_u = input("Enter the grasp point on the object u: ")
        input_grasp_point_v = input("Enter the grasp point on the object v: ")
        center = np.array([int(input_grasp_point_u), int(input_grasp_point_v)])

        suction_directions = normals[int(center[1]), int(center[0]), :]
        # euler_angle = [math.asin(suction_directions[0]), math.asin(
        #     suction_directions[1])]
        euler_angle = [0.0, 0.0]

        # sampled suctions
        # origin is at top left side, so calculate as per the origin
        print("center ", center)
        # depth /= 1000.0
        center3D_depth_value = depth[int(center[1]), int(center[0])]

        pointz = center3D_depth_value
        pointx = (center[0] - cx) * pointz / fx
        pointy = (center[1] - cy) * pointz / fy
        center_3D = (pointx, pointy, pointz)

        print("center ", center_3D)
        score_image = np.zeros_like(heatmap)
        best_distance = sys.maxsize
        best_point = [pointx, pointy, pointz]

        self.drawGaussian(score_image, [int(center[0]), int(center[1])], 1.0, 3)

        score_image *= 255
        score_image = score_image.clip(0, 255)
        score_image = score_image.astype(np.uint8)
        score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
        rgb_image = rgb_img
        rgb_image = 0.5 * rgb_image + 0.5 * score_image
        rgb_image = rgb_image.astype(np.uint8)
        # cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/grasp2.png", rgb_image)
        cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/"+str(time_stamp)+"grasp2.png", rgb_image)
        # im = Image.fromarray(rgb_image)

        # visu_dir = save_root
        # os.makedirs(visu_dir, exist_ok=True)
        # im.save(visu_dir+'/%04d_sampled' % count+'.png')
        # count += 1
        # cv2.destroyAllWindows()
        # best_point_return = [best_point[0], best_point[1], 1.081]
        best_point_return = [float(best_point[0]), float(best_point[1]), float(best_point[2])]


        return best_point_return, euler_angle, center

# if __name__ == "__main__":
#     inference()
