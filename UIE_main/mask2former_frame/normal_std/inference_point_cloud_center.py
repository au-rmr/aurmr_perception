from normal_std.policy import estimate_suction
from mpl_toolkits import mplot3d
from demo.segnetv2_demo import SegnetV2
import os
import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scipy.io as scio
# from policy import create_point_cloud_from_depth_image
import math
import matplotlib.pyplot as plt
import csv

import sys
sys.path.append(
    "/home/soofiyan_ws_force_calibration_suction_point/workspaces/aurmr_perception/UIE_main/mask2former_frame/")
split = 'test_seen'
camera = 'kinect'
# save_root = '/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/UIE_main/result'


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

    # Function converts pixel coordinate given depth of that pixel to wolrd coordinate
    def convert_pixel_to_point_cloud(self, depth_point, rgb_point, camera_info):
        point_z = depth_point
        point_x = (rgb_point[0] - camera_info.cx) * point_z / camera_info.fx
        point_y = (rgb_point[1] - camera_info.cy) * point_z / camera_info.fy
        return np.array([point_x, point_y, point_z])

    # Function to convert the depth iamge to point cloud matrix
    def convert_rgb_depth_to_point_cloud(self, depth_img, camera_info):
        xmap = np.arange(camera_info.width)
        ymap = np.arange(camera_info.height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth_img
        points_x = (xmap - camera_info.cx) * points_z / camera_info.fx
        points_y = (ymap - camera_info.cy) * points_z / camera_info.fy
        return np.array([points_x, points_y, points_z])

    # Function to convert the world coordinate to pixel coordinate
    def convert_point_cloud_to_pixel(self, depth_point, camera_info):
        point_z = depth_point[2]
        u = (depth_point[0]*camera_info.fx)/point_z + camera_info.cx
        v = (depth_point[1]*camera_info.fy)/point_z + camera_info.cy
        return np.array([u, v])

    # Function for reprojection from the depth coordinate to pixel coordinate without depth
    # Here the depth x and y coordinate is searched in the point cloud which gives the closest x and y coordinate along with the depth value
    def find_nearest(self, array, value, camera_info):
        diff = 999.
        point_cloud_index = np.array([0, 0])
        # point_cloud_index = np.array([0, 0, 0])
        # diff_array_x = np.abs(np.subtract(
        #     array[:1, :, :], value[0]), dtype=np.float32)
        # diff_array_y = np.abs(np.subtract(
        #     array[1:2, :, :], value[1]), dtype=np.float32)
        # # print("unravel method", np.unravel_index(np.add(diff_array_x, diff_array_y).argmin(), np.add(diff_array_x, diff_array_y).shape))
        # # point cloud index comes as this channel, column index, row index
        # point_cloud_index = np.unravel_index(np.add(
        #     diff_array_x, diff_array_y).argmin(), np.add(diff_array_x, diff_array_y).shape)
        for j in range(camera_info.height):
            for k in range(camera_info.width):
                compare = abs(array[0][j][k]-value[0]) + abs(array[1][j][k]-value[1])
                if(diff > compare):
                    diff = compare
                    point_cloud_index = np.array([k, j])
        # print("using for loop", point_cloud_index)
        # return np.array([point_cloud_index[2], point_cloud_index[1]])
        return point_cloud_index

    def inference(self, rgb_img, depth_img, mask, bin_id):
        '''
        For grasping we need the centroid of the object with respect to 1920x1080 resolution
        '''
        depth = depth_img.astype(np.float32)/1000.0
        print('depth shape', depth.shape)
        seg_mask1 = mask.astype(np.uint8)
        print("shape of rgb", rgb_img.shape)
        # applying contour to get the edges of the mask
        contours = cv2.findContours(
            seg_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        rotrect = cv2.minAreaRect(big_contour)
        (center), (width, height), angle = rotrect
        print("center 4k", center)
        center3D_depth_value = depth[int(center[1]), int(center[0])]
        fx, fy = 909.4390869140625, 909.466796875
        cx, cy = 960.0811157226562, 546.9479370117188
        width = 1920
        height = 1080
        pointz = center3D_depth_value
        pointx = (center[0] - cx) * pointz / fx
        pointy = (center[1] - cy) * pointz / fy
        center_3D = (pointx, pointy, pointz)
        best_point_return = [center_3D[0], center_3D[1], 1.081]
        euler_angle = [0.0, 0.0]


        s = 1000.0
        camera_info = CameraInfo(width, height, fx, fy, cx, cy, s)

        heatmap, normals, point_cloud = estimate_suction(
            depth, seg_mask1, camera_info)

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
        rgb_image = rgb_img
        rgb_image = 0.5 * rgb_image + 0.5 * score_image
        rgb_image = rgb_image.astype(np.uint8)
        cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/grasp1.png", rgb_image)
        # im = Image.fromarray(rgb_image)

        # sampled suctions
        # origin is at top left side, so calculate as per the origin
        print("center ", center)
        center3D_depth_value = depth[int(center[1]), int(center[0])]

        pointz = center3D_depth_value
        pointx = (center[0] - cx) * pointz / fx
        pointy = (center[1] - cy) * pointz / fy
        center_3D = (pointx, pointy, pointz)

        print("center ", center_3D)
        score_image = np.zeros_like(heatmap)
        best_distance = sys.maxsize
        suction_number = 0
        best_point = [pointx, pointy, pointz]

        for i in range(suction_scores.shape[0]):
            center3D_depth_value = depth[int(idx0[i]), int(idx1[i])]

            pointz = center3D_depth_value
            pointx = (idx1[i] - cx) * pointz / fx
            pointy = (idx0[i] - cy) * pointz / fy
            # print(pointx, pointy, pointz)
            # printing all the suction points
            # print("all suction points ", pointx, pointy, pointz)
            distance = math.sqrt(
                (center_3D[0] - pointx)**2 + (center_3D[1] - pointy)**2)
            if(distance < best_distance):
                suction_number = i
                best_distance = distance
                best_point = [pointx, pointy, pointz]

        if(depth[(idx0[suction_number]), int(idx1[suction_number])] < 1.3):
            self.drawGaussian(score_image, [
                idx1[suction_number], idx0[suction_number]], 1.0, 3)
            self.drawGaussian(
                score_image, [int(center[0]), int(center[1])], 0.5, 3)
        euler_angle = [math.asin(suction_directions[suction_number][0]), math.asin(
            suction_directions[suction_number][1])]
        # print(center_3D)

        # print("Rotation Matrix ", )
        # print("directions", suction_directions[suction_number])
        print("direction angles x, y ", math.asin(
            suction_directions[suction_number][0])*180/math.pi, math.asin(
            suction_directions[suction_number][1])*180/math.pi)
        # print("translations", suction_translations[suction_number])

        score_image *= 255
        score_image = score_image.clip(0, 255)
        score_image = score_image.astype(np.uint8)
        score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
        rgb_image = rgb_img
        rgb_image = 0.5 * rgb_image + 0.5 * score_image
        rgb_image = rgb_image.astype(np.uint8)
        

        best_point_return = [best_point[0], best_point[1], 1.081]

        rgb_image = rgb_img.copy()
        cv2.circle(rgb_image, (int(round(center[0])), int(round(center[1]))), 1, (0, 0, 255), 5)
        cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/grasp2.png", rgb_image)

        image_crop_sizes = np.array([[295*2, 167*2, 381*2, 219*2],
                                    [381*2, 167*2, 483*2, 215*2],
                                    [483*2, 167*2, 585*2, 214*2],
                                    [585*2, 163*2, 673*2, 210*2],
                                    [296*2, 232*2, 384*2, 272*2],
                                    [382*2, 232*2, 485*2, 268*2],
                                    [485*2, 232*2, 586*2, 268*2],
                                    [586*2, 232*2, 674*2, 263*2],
                                    [298*2, 286*2, 387*2, 370*2],
                                    [384*2, 286*2, 487*2, 365*2],
                                    [485*2, 284*2, 587*2, 366*2],
                                    [586*2, 283*2, 677*2, 362*2],
                                    [299*2, 384*2, 387*2, 422*2],
                                    [387*2, 387*2, 489*2, 417*2],
                                    [487*2, 383*2, 589*2, 417*2],
                                    [587*2, 381*2, 678*2, 414*2]], dtype=int)

        # bin_bounds = {
#           '1H':[295*2, 167*2, 381*2, 234*2],
#           '2H':[381*2, 167*2, 483*2, 233*2],
#           '3H':[483*2, 167*2, 585*2, 233*2],
#           '4H':[585*2, 163*2, 673*2, 228*2],
#           '1G':[296*2, 232*2, 384*2, 286*2],
#           '2G':[382*2, 232*2, 485*2, 286*2],
#           '3G':[485*2, 232*2, 586*2, 284*2],
#           '4G':[586*2, 232*2, 674*2, 283*2],
#           '1F':[298*2, 286*2, 387*2, 387*2],
#           '2F':[384*2, 286*2, 487*2, 383*2],
#           '3F':[485*2, 284*2, 587*2, 381*2],
#           '4F':[586*2, 283*2, 677*2, 377*2],
#           '1E':[299*2, 384*2, 387*2, 437*2],
#           '2E':[387*2, 387*2, 489*2, 433*2],
#           '3E':[487*2, 383*2, 589*2, 433*2],
#           '4E':[587*2, 381*2, 678*2, 428*2],
#             }
        image_bin = ["1H", "2H", "3H", "4H", "1G", "2G", "3G", "4G", "1F", "2F", "3F", "4F", "1E", "2E", "3E", "4E"]

        count = 0
        image_bin_index = image_bin.index(bin_id)
        image_crop_ind = image_crop_sizes[image_bin_index]
        depth_img = depth_img.copy()
        depth = depth_img.astype(np.float32)/1000.0
        depth = depth[image_crop_ind[1]:image_crop_ind[3], image_crop_ind[0]:image_crop_ind[2]]
        # depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)
        print("Depth Image", np.mean(depth), " ",
              np.max(depth), " ", np.min(depth))
        print(depth.shape)
        seg_mask = mask.copy()
        seg_mask = seg_mask[image_crop_ind[1]:image_crop_ind[3], image_crop_ind[0]:image_crop_ind[2]]
        
        # seg_mask = cv2.resize(seg_mask, (640, 480), interpolation=cv2.INTER_AREA)
        seg_mask = np.asarray((seg_mask == 1), dtype=np.uint8)
        # seg_mask = temp_mask*(mask == 1)
        rgb_img = rgb_img.copy()
        rgb_img = rgb_img[image_crop_ind[1]:image_crop_ind[3], image_crop_ind[0]:image_crop_ind[2]]
        # rgb_img = cv2.resize(rgb_img, (640, 480), interpolation=cv2.INTER_AREA)
        
        
        cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/resized_cropped_mask.png", seg_mask)
        # applying contour to get the edges of the mask

        # For RGB
        # fx, fy = 1940.1367*(640/(image_crop_ind[2]-image_crop_ind[0])), 1940.1958*(480/(image_crop_ind[3]-image_crop_ind[1]))
        fx, fy = 909.4390869140625, 909.466796875
        # For Depth
        # fx, fy = 504.9533, 504.976
        # cx, cy = 514.2323, 507.22818

        # These values are for Image with dimension 4096x3072
        
        width = (image_crop_ind[2]-image_crop_ind[0])
        height = (image_crop_ind[3]-image_crop_ind[1])
        cx, cy = width/2, height/2

        s = 1000.0
        camera_info = CameraInfo(width, height, fx, fy, cx, cy, s)

        # applying contour to get the edges of the mask
        # contours = cv2.findContours(
        #     seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = contours[0] if len(contours) == 2 else contours[1]
        # big_contour = max(contours, key=cv2.contourArea)

        # # get rotated rectangle from contour
        # # get its dimensions
        # # get angle relative to horizontal from rotated rectangle
        # rotrect = cv2.minAreaRect(big_contour)
        # (center), (width, height), angle = rotrect
        # This is the center obtained through the coutnout function and this is pixel of the segmentation mask
        center = np.array([int(round(center[0] - image_crop_ind[0])), int(round(center[1] - image_crop_ind[1]))])
        print("center", center)
        
        print(depth[center[1], center[0]])
        centroid = np.array([center[0], center[1]])
        '''
        This is the depth value of the centroid which was obtained through mask
        Here the depth image reads in reverse order i.e. the column and row are inverted
        '''
        center_depth = depth[center[1], center[0]]
        '''
        Here the centroid of the object is converted to world coordinate
        '''
        object_centroid = self.convert_pixel_to_point_cloud(
            center_depth, center, camera_info)
        # print(object_centroid)
        '''
        The z coordinate of object centroid is of no use because in reprojection this depth value is not useful
        '''
        object_centroid[1] += 0.02
        object_centroid[2] = 0.0
        '''
        Base coordinate array is the radius of suction cup and this is basically the right point of the suction cup
        '''
        base_coordinate = np.array([0.02, 0, 0], dtype=np.float32)
        suction_coordinates = [base_coordinate]
        '''
        Adding the right point of the suction cup to the object centroid as we want to obtain all the points around the object centroid
        '''
        object_base_coordinate = object_centroid + base_coordinate
        object_suction_coordinate = [object_base_coordinate]

        '''
        This loop is used to obtain all the 8 points of the suction cup by rotating the first vector
        '''
        for angle in range(45, 360, 45):
            x = base_coordinate[0]*math.cos(angle*math.pi/180) - \
                base_coordinate[1]*math.sin(angle*math.pi/180)
            y = base_coordinate[0]*math.sin(angle*math.pi/180) + \
                base_coordinate[1]*math.cos(angle*math.pi/180)
            '''
            Appending all the coordiantes in suction_cooridnates and the object_suction_coordinate is the x and y 3D cooridnate of the object suction points
            '''
            suction_coordinates = np.append(
                suction_coordinates, np.array([[x, y, 0.]]), axis=0)
            object_suction_coordinate = np.append(object_suction_coordinate, np.array(
                [[x+object_centroid[0], y+object_centroid[1], 0.]]), axis=0)

        '''
        Here we convert the depth image to point cloud and the format of the point cloud is channels, height and width
        '''
        point_cloud = self.convert_rgb_depth_to_point_cloud(depth, camera_info)

        center_transform = self.find_nearest(point_cloud, np.array([object_centroid[0], object_centroid[1]]), camera_info)
        cv2.circle(rgb_img, (int(round(center_transform[0])), int(
            round(center_transform[1]))), 1, (0, 0, 255), 5)
        print("center transform", center_transform)
        print(depth[center_transform[1], center_transform[0]])

        suction_projections = np.empty((0, 3), float)
        for suction_points in suction_coordinates:
            suction_point_mark = self.find_nearest(point_cloud, np.array([object_centroid[0]+suction_points[0], object_centroid[1]+suction_points[1]]), camera_info)
            # print("point_cloud cooridnates", self.find_nearest(point_cloud, np.array([object_centroid[0]+suction_points[0], object_centroid[1]+suction_points[1]]), camera_info))
            '''
            here for each desired suction point coordinate which we saved in the suction_cooridantes is used to find the nearest cooridnate of the suction cup
            This object_centroid[0]+suction_point is same as cooridnates appended in the object_suction_coorindates
            '''
            suction_point = self.find_nearest(point_cloud, [
                                              object_centroid[0]+suction_points[0], object_centroid[1]+suction_points[1]], camera_info)
            cv2.circle(rgb_img, (int(round(suction_point_mark[0])), int(
                round(suction_point_mark[1]))), 1, (255, 0, 0), 5)
            # print(int(round(suction_point[0])), int(round(suction_point[1])))
            # print("desired", object_centroid[0]+suction_points[0], object_centroid[1]+suction_points[1])
            # print("actual", point_cloud[0][suction_point[1]][suction_point[0]], point_cloud[1][suction_point[1]][suction_point[0]])
            '''
            Now the suction_point is basically the actual point on the poit cloud whihc is the closest point to the desired coordinates
            Now after getting the pixel cooridnates, here the world cooridnates of each point is getting appended in the suction_projections 
            '''
            suction_projections = np.vstack((suction_projections, np.array(
                [point_cloud[0][suction_point[1]][suction_point[0]], point_cloud[1][suction_point[1]][suction_point[0]], depth[suction_point[1], suction_point[0]]])))

        # for suction in suction_projections:
        #     print(self.convert_point_cloud_to_pixel(suction, camera_info))
        transformed_suction_projections = np.empty((0, 3), float)
        if(euler_angle[0] > 20*math.pi/180 or euler_angle[0] < -20*math.pi/180):
            euler_angle[0] = 0.0
        if(euler_angle[1] > 20*math.pi/180 or euler_angle[1] < -10*math.pi/180):
            euler_angle[1] = 0.0
        # euler_angle[0] = 0.0
        # euler_angle[1] = 0.0
        x_axis_angle = euler_angle[1]*180/math.pi
        y_axis_angle = euler_angle[0]*180/math.pi
        
        print(y_axis_angle, x_axis_angle)
        '''
        These are the 3D rotation matrix around each axis 
        '''
        R_y = np.array([[math.cos(math.pi*y_axis_angle/180), 0, math.sin(math.pi*y_axis_angle/180)],
                       [0, 1, 0], [-math.sin(math.pi*y_axis_angle/180), 0, math.cos(math.pi*y_axis_angle/180)]])
        R_x = np.array([[1,0,0], [0, math.cos(math.pi*x_axis_angle/180), -math.sin(math.pi*x_axis_angle/180)],
                       [0, -math.sin(math.pi*x_axis_angle/180), math.cos(math.pi*x_axis_angle/180)]])
        R_xy = R_y*R_x
        '''
        These are the trasnformed cooridnates when the suction cup is rotated at an angle around the obejct centroid
        '''
        for suction in suction_coordinates:
            '''
            These are the base coordinates of suction which are rotated with respect to origin (0,0,0)
            '''
            # suction[0] = suction[0]+0.003
            transformed_base_pose1 = np.dot(R_x, np.transpose(suction))
            transformed_base_pose2 = np.dot(R_y, np.transpose(suction))
            # print("deth change", transformed_base_pose1[2]+transformed_base_pose2[2])
            transformed_base_pose = np.array([transformed_base_pose2[0], transformed_base_pose1[1], transformed_base_pose1[2]+transformed_base_pose2[2]])
            
            # transformed_base_pose = np.dot(R_xy, np.transpose(suction))
            print("without transformation in loop", suction)
            print("with transformation in loop", transformed_base_pose)
            # print(suction)
            # print(transformed_base_pose)
            '''
            Again searching for the closest cooridnate for that particular x adnd y coordinate
            '''
            suction_point = self.find_nearest(point_cloud, [
                                              object_centroid[0]+transformed_base_pose[0], object_centroid[1]+transformed_base_pose[1]], camera_info)
            '''
            Stacking the actual trasnformed suction point coordiante with respect to the object centroid in the transformed_suction_projections array
            '''
            cv2.circle(rgb_img, (int(round(suction_point[0])), int(round(suction_point[1]))), 1, (0, 255, 0), 5)
            transformed_suction_projections = np.vstack((transformed_suction_projections, np.array([point_cloud[0][suction_point[1]][suction_point[0]],
                                                                                                    point_cloud[1][suction_point[1]][suction_point[0]], depth[suction_point[1], suction_point[0]]+transformed_base_pose[2]])))
            # transformed_suction_projections = np.vstack((transformed_suction_projections, R_y*suction))
            # print("appending", transformed_suction_projections)
        
        # seconds = time.time()
        # local_time = str(time.ctime(seconds))
        print("without transformation", suction_projections)
        print("with transformations", transformed_suction_projections)

        minimum_suction_point = np.min(suction_projections[:,2])
        ri = np.array([])
        for i in range(len(suction_projections)):
            value = min(1, np.abs(
                (suction_projections[i][2]-minimum_suction_point)*math.cos(x_axis_angle*math.pi/180.0)*math.cos(y_axis_angle*math.pi/180.0))/(0.023))
            ri = np.append(ri, value)

        
        print(ri)
        score_wo_trasnform = 1-np.max(ri)
        print("Conical Springs score without transformation",score_wo_trasnform)

        minimum_suction_point = np.min(transformed_suction_projections[:,2])
        ri = np.array([])
        for i in range(len(transformed_suction_projections)):
            value = min(1, np.abs(
                (transformed_suction_projections[i][2]-minimum_suction_point)*math.cos(x_axis_angle*math.pi/180.0)*math.cos(y_axis_angle*math.pi/180.0))/(0.023))
            ri = np.append(ri, value)
        
        print(ri)
        score_w_trasnform = 1-np.max(ri)
        print("Conical Springs score with transformation",score_w_trasnform)
        
        cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/rgb_suction_points.png", rgb_img)

        name_object = input("name of the object: ")
        save_bool = input("Do you want to save these files in proper data or improper data? Type 1:proper or 2:improper or 0:discard --> ")
        if(int(save_bool) == 1):
            local_time = time.strftime("%Y%m%d-%H%M%S")
            try:
                os.makedirs("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Proper_Data/"+str(name_object)+"/"+str(local_time))
            except:
                pass
            my_dict = {"suction deformation scor without trasnformation": score_wo_trasnform, "suction deformation scor with trasnformation": score_w_trasnform, "center_pixel": center}
            cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Proper_Data/"+str(name_object)+"/"+str(local_time)+"/rgb.png", rgb_img)
            np.save("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Proper_Data/"+str(name_object)+"/"+str(local_time)+"/depth.npy", depth)
            cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Proper_Data/"+str(name_object)+"/"+str(local_time)+"/mask.png", seg_mask)
            with open('/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Proper_Data/'+str(name_object)+"/"+str(local_time)+'/suction_properties.csv', 'w') as f:
                w = csv.DictWriter(f, my_dict.keys())
                w.writeheader()
                w.writerow(my_dict)
        elif(int(save_bool) == 2):
            local_time = time.strftime("%Y%m%d-%H%M%S")
            try:
                os.makedirs("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Improper_Data/"+str(name_object)+"/"+str(local_time))
            except:
                pass
            my_dict = {"suction deformation scor without trasnformation": score_wo_trasnform, "suction deformation scor with trasnformation": score_w_trasnform, "center_pixel": center}
            cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Improper_Data/"+str(name_object)+"/"+str(local_time)+"/rgb.png", rgb_img)
            np.save("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Improper_Data/"+str(name_object)+"/"+str(local_time)+"/depth.npy", depth)
            cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Improper_Data/"+str(name_object)+"/"+str(local_time)+"/mask.png", seg_mask)
            with open('/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Improper_Data/'+str(name_object)+"/"+str(local_time)+'/suction_properties.csv', 'w') as f:
                w = csv.DictWriter(f, my_dict.keys())
                w.writeheader()
                w.writerow(my_dict)
        elif(int(save_bool) == 0):
            local_time = time.strftime("%Y%m%d-%H%M%S")
            try:
                os.makedirs("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Trash/"+str(name_object)+"/"+str(local_time))
            except:
                pass
            my_dict = {"suction deformation scor without trasnformation": score_wo_trasnform, "suction deformation scor with trasnformation": score_w_trasnform, "center_pixel": center}
            cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Trash/"+str(name_object)+"/"+str(local_time)+"/rgb.png", rgb_img)
            np.save("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Trash/"+str(name_object)+"/"+str(local_time)+"/depth.npy", depth)
            cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Trash/"+str(name_object)+"/"+str(local_time)+"/mask.png", seg_mask)
            with open('/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Data_Collection_Suction_Cup_Deformation/Trash/'+str(name_object)+"/"+str(local_time)+'/suction_properties.csv', 'w') as f:
                w = csv.DictWriter(f, my_dict.keys())
                w.writeheader()
                w.writerow(my_dict)

        ri = np.array([])
        undeformed_l = math.sqrt((suction_coordinates[0][0] - suction_coordinates[2][0])**2 + (suction_coordinates[0][1] - suction_coordinates[2][1])**2 + (1.081 - 1.081)**2)
        for i in range(len(transformed_suction_projections)):
            deformed_l = math.sqrt((transformed_suction_projections[i][0] - transformed_suction_projections[(i+2)%8][0])**2 + (transformed_suction_projections[i][1] - transformed_suction_projections[(i+2)%8][1])**2 + (transformed_suction_projections[i][2] - transformed_suction_projections[(i+2)%8][2])**2)
            ri = np.append(ri, min(1, np.abs((deformed_l - undeformed_l)/(undeformed_l))))
        
        # print(ri)
        score = 1-np.max(ri)
        print("Perimter Springs score ", score)
        # seconds = time.time()
        # local_time = time.ctime(seconds)
        # with open('/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/Mask_Results/Suction_Cup_Points_Data/suction_point_score.csv', mode='w') as suction_file:
        #     employee_writer = csv.writer(suction_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        #     employee_writer.writerow([local_time, ])
        #     employee_writer.writerow(['Erica Meyers', 'IT', 'March'])

        # print(self.convert_pixel_to_point_cloud(depth[265, 290], np.array([265, 290]), camera_info))
        # print(object_centroid[0]+0.02, object_centroid[1])

        # print(suction_coordinates)
        # print(object_suction_coordinate)
        euler_angle[0] = 0.0
        euler_angle[1] = 0.0
        return best_point_return, euler_angle


# if __name__ == "__main__":
    # input_image_rgb = cv2.imread(
    #     "/home/soofiyan_ws_force_calibration_suction_point/workspaces/aurmr_perception/UIE_main/data_full_pod/Dataset/rgbImages/Kinect/Cropped_RGB/crop_rgb_3H.png")
    # input_image_depth = np.load(
    #     "/home/soofiyan_ws_force_calibration_suction_point/workspaces/aurmr_perception/UIE_main/data_full_pod/Dataset/depthImages/Kinect/Crop_depth_numpy/crop_depth_3H.npy")

    # object = SegnetV2()
    # masks, mask_crop = object.mask_generator(input_image_rgb)

    # object = run_normal_std()
    # object.inference(input_image_rgb, input_image_depth, mask_crop)
