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
from mask2former_frame.demo.segnetv2_demo import SegnetV2
######################################################################################################
object = SegnetV2()
masks = object.mask_generator(
    "/home/soofiyanatar/Documents/AmazonHUB/UIE_main/annotated_real_v1_resized/images/scene_03/bin_1E/bin_1E_color_0006.png")
######################################################################################################


split = 'test_seen'
camera = 'kinect'
save_root = '/home/soofiyanatar/Documents/AmazonHUB/suctionnet-baseline/normal_std/Results'


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
        # print('x:', x.shape)
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

    def inference(self):
        count = 0
        for mask in masks:
            # reading the depth mask and rgb file, also reading the meta file for intrinsic parameters
            # segmask_file = "/home/soofiyanatar/Documents/AmazonHUB/UIE-main/masks/label0.png"
            depth_file = "/home/soofiyanatar/datasets/Full_Dataset/depth_2.png"
            rgb_file = "/home/soofiyanatar/Documents/AmazonHUB/UIE_main/annotated_real_v1_resized/images/scene_03/bin_1E/bin_1E_color_0006.png"

            # segmask_file = "/home/soofiyanatar/datasets/Full_Dataset/label_segnet.png"
            # depth_file = "/home/soofiyanatar/datasets/Full_Dataset/depth_segnet.png"
            # rgb_file = "/home/soofiyanatar/datasets/Full_Dataset/rgb_segnet.png"

            # Getting the depth image in meters and segmentation mask in its original state (grayscale)
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(
                np.float32)/1000.0
            seg_mask = mask.astype(np.uint8)

            # applying contour to get the edges of the mask
            contours = cv2.findContours(
                seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            big_contour = max(contours, key=cv2.contourArea)

            # get rotated rectangle from contour
            # get its dimensions
            # get angle relative to horizontal from rotated rectangle
            rotrect = cv2.minAreaRect(big_contour)
            (center), (width, height), angle = rotrect
            box = cv2.boxPoints(rotrect)
            boxpts = np.int0(box)

            # draw rotated rectangle on copy of input
            rotrect_img = seg_mask.copy()

            # correcting the angle
            if angle < -45:
                angle = -(90 + angle)

            # otherwise, check width vs height
            else:
                if width > height:
                    angle = -(90 + angle)
                else:
                    angle = -angle

            # adding condition on different angles on height and weight
            if((angle <= -70 and angle >= -160) or (angle >= 70 and angle <= 160)):
                angle += 90
                boxpts[0][1] += (height/1.5)*math.cos((angle)*math.pi/180)
                boxpts[1][1] += (height/1.5)*math.cos((angle)*math.pi/180)
                boxpts[0][0] += (height/1.5)*math.sin((angle)*math.pi/180)
                boxpts[1][0] += (height/1.5)*math.sin((angle)*math.pi/180)
            elif((angle < -160) or (angle > 160)):
                boxpts[0][1] += (height*1.25)*math.cos((180+angle)*math.pi/180)
                boxpts[1][1] += (height*1.25)*math.cos((180+angle)*math.pi/180)
                boxpts[0][0] += (height*1.25)*math.sin((180+angle)*math.pi/180)
                boxpts[1][0] += (height*1.25)*math.sin((180+angle)*math.pi/180)
            else:
                boxpts[1][1] += (height/1.5)*math.cos((angle)*math.pi/180)
                boxpts[2][1] += (height/1.5)*math.cos((angle)*math.pi/180)
                boxpts[1][0] += (height/1.5)*math.sin((angle)*math.pi/180)
                boxpts[2][0] += (height/1.5)*math.sin((angle)*math.pi/180)

            # getting rotation matrix on the basis of angle
            M = cv2.getRotationMatrix2D(center, -angle, scale=1.0)

            # after reducing the points we have to mask that area with black for proper detection
            cv2.drawContours(rotrect_img, [boxpts], 0, (255, 255, 255), 1)

            # Now for visualization we resized the image
            rotrect_img = cv2.resize(rotrect_img, (1024, 768),
                                     interpolation=cv2.INTER_LINEAR)

            # fill the polygon in segmentation mask using teje points
            points = np.array([[boxpts[0][0], boxpts[0][1]], [boxpts[1][0], boxpts[1][1]], [
                boxpts[2][0], boxpts[2][1]], [boxpts[3][0], boxpts[3][1]]])
            seg_mask = cv2.fillPoly(seg_mask, np.int32([points]), (0, 0, 0))

            (thresh, binary_mask) = cv2.threshold(
                seg_mask, 1, 255, cv2.THRESH_BINARY)
            binary_mask = Image.fromarray(binary_mask)
            binary_mask.save(save_root+'/%04d_example' % count+'.png')

            kernel = np.ones((3, 3), np.uint8)
            seg_mask = cv2.erode(seg_mask, kernel, iterations=1)
            seg_mask = cv2.dilate(seg_mask, kernel, iterations=1)

            # For RGB
            fx, fy = 1940.1367, 1940.1958
            # For Depth
            # fx, fy = 504.9533, 504.976
            # cx, cy = 514.2323, 507.22818

            # These values are for Image with dimension 4096x3072
            # cx, cy = 2048.7397, 1551.3889
            # width = 4096
            # height = 3072

            # These values are for Image with dimension 360x160
            # cx, cy = 180.0646, 80.8015
            # width = 360
            # height = 160

            # These values are for Image with dimension 450x320
            cx, cy = 225.08075, 161.603
            width = 450
            height = 320

            s = 1000.0
            camera_info = CameraInfo(width, height, fx, fy, cx, cy, s)

            heatmap, normals, point_cloud = estimate_suction(
                depth, seg_mask, camera_info)

            k_size = 15
            kernel = self.uniform_kernel(k_size)
            kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
            heatmap = np.pad(heatmap, k_size//2)
            heatmap = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
            heatmap = F.conv2d(heatmap, kernel).squeeze().numpy()

            suction_scores, idx0, idx1 = self.grid_sample(
                heatmap, down_rate=15, topk=50)
            suction_directions = normals[idx0, idx1, :]
            suction_translations = point_cloud[idx0, idx1, :]

            # pridictions
            score_image = heatmap
            score_image *= 255

            score_image = score_image.clip(0, 255)
            score_image = score_image.astype(np.uint8)
            score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
            rgb_image = np.array(Image.open(rgb_file), dtype=np.float32)
            rgb_image = 0.5 * rgb_image + 0.5 * score_image
            rgb_image = rgb_image.astype(np.uint8)
            im = Image.fromarray(rgb_image)

            # For visulaizing using cv2 in realtime
            # temp_rgb = cv2.resize(rgb_image, (1280, 720),
            #                       interpolation=cv2.INTER_AREA)
            # cv2.imshow("rgb ", rgb_image)
            # cv2.waitKey(0)

            visu_dir = save_root
            os.makedirs(visu_dir, exist_ok=True)
            im.save(visu_dir+'/%04d' % count+'.png')

            # sampled suctions
            # origin is at top left side, so calculate as per the origin
            center3D_depth_value = depth[int(center[1]), int(center[0])]

            pointz = center3D_depth_value
            pointx = (center[1] - cx) * pointz / fx
            pointy = (center[0] - cy) * pointz / fy
            center_3D = (pointx, pointy, pointz)

            score_image = np.zeros_like(heatmap)
            best_distance = sys.maxsize
            suction_number = 0

            for i in range(suction_scores.shape[0]):
                center3D_depth_value = depth[int(idx0[i]), int(idx1[i])]

                pointz = center3D_depth_value
                pointx = (idx0[i] - cx) * pointz / fx
                pointy = (idx1[i] - cy) * pointz / fy

                # printing all the suction points
                # print("all suction points ", pointx, pointy, pointz)
                distance = math.sqrt(
                    (center_3D[0] - pointx)**2 + (center_3D[1] - pointy)**2)
                if(distance < best_distance and suction_scores[i] > 0.4 and pointz < 1.2):
                    suction_number = i
                    best_distance = distance

            if(depth[(idx0[suction_number]), int(idx1[suction_number])] < 1.2):
                self.drawGaussian(score_image, [
                    idx1[suction_number], idx0[suction_number]], 1.0, 3)
                self.drawGaussian(
                    score_image, [int(center[0]), int(center[1])], 0.5, 3)

            print(center_3D)

            print("Rotation Matrix ", )
            print("directions", suction_directions[suction_number])
            print("direction angles x, y ", math.asin(
                suction_directions[suction_number][0])*180/math.pi, math.asin(
                suction_directions[suction_number][1])*180/math.pi)
            print("translations", suction_translations[suction_number])

            score_image *= 255
            score_image = score_image.clip(0, 255)
            score_image = score_image.astype(np.uint8)
            score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
            rgb_image = np.array(Image.open(rgb_file), dtype=np.float32)
            rgb_image = 0.5 * rgb_image + 0.5 * score_image
            rgb_image = rgb_image.astype(np.uint8)
            im = Image.fromarray(rgb_image)

            visu_dir = save_root
            os.makedirs(visu_dir, exist_ok=True)
            im.save(visu_dir+'/%04d_sampled' % count+'.png')
            count += 1
        cv2.destroyAllWindows()


# if __name__ == "__main__":

    # inference()
