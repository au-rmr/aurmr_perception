import sys
sys.path.append(
    "/home/soofiyan_ws/workspaces/aurmr_perception/UIE_main/mask2former_frame/")
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
from policy import create_point_cloud_from_depth_image
import math
import matplotlib.pyplot as plt


split = 'test_seen'
camera = 'kinect'
# save_root = '/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/UIE_main/result'


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
        point_cloud_index = np.array([0, 0, 0])
        diff_array_x = np.abs(np.subtract(
            array[:1, :, :], value[0]), dtype=np.float32)
        diff_array_y = np.abs(np.subtract(
            array[1:2, :, :], value[1]), dtype=np.float32)
        # print("unravel method", np.unravel_index(np.add(diff_array_x, diff_array_y).argmin(), np.add(diff_array_x, diff_array_y).shape))
        # point cloud index comes as this cahnnel, column index, row index
        point_cloud_index = np.unravel_index(np.add(
            diff_array_x, diff_array_y).argmin(), np.add(diff_array_x, diff_array_y).shape)
        # for j in range(camera_info.height):
        #     for k in range(camera_info.width):
        #         compare = abs(array[0][j][k]-value[0]) + abs(array[1][j][k]-value[1])
        #         if(diff > compare):
        #             diff = compare
        #             point_cloud_index = np.array([k, j])
        # print("using for loop", np.array([point_cloud_index[2], point_cloud_index[1]]))
        return np.array([point_cloud_index[2], point_cloud_index[1]])

    def inference(self, rgb_img, depth_img, mask):
        count = 0
        depth = depth_img.astype(np.float32)
        print("Depth Image", np.mean(depth), " ",
              np.max(depth), " ", np.min(depth))
        seg_mask = mask.astype(np.uint8)
        seg_mask = np.asarray((mask == 2), dtype=np.uint8)
        # seg_mask = temp_mask*(mask == 1)
        print(seg_mask.shape)
        # applying contour to get the edges of the ma

        # For RGB

        fx, fy = 1940.1367*(640/412), 1940.1958*(480/268)
        # For Depth
        # fx, fy = 504.9533, 504.976
        # cx, cy = 514.2323, 507.22818

        # These values are for Image with dimension 4096x3072
        cx, cy = 320, 240
        width = 640
        height = 480

        s = 1000.0
        camera_info = CameraInfo(width, height, fx, fy, cx, cy, s)

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
        # This is the center obtained through the coutnout function and this is pixel of the segmentation mask
        center = np.array([int(round(center[0])), int(round(center[1]))])
        print(center)
        cv2.circle(rgb_img, (int(round(center[0])), int(
            round(center[1]))), 1, (0, 0, 255), 5)
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
        print(object_centroid)
        '''
        The z coordinate of object centroid is of no use because in reprojection this depth value is not useful
        '''
        object_centroid[2] = 0.0

        '''
        Base coordinate array is the radius of suction cup and this is basically the right point of the suction cup
        '''
        base_coordinate = np.array([0.0205, 0, 0], dtype=np.float32)
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
        suction_projections = np.empty((0, 3), float)
        for suction_points in suction_coordinates:
            # print("point_cloud cooridnates", self.find_nearest(point_cloud, np.array([object_centroid[0]+suction_points[0], object_centroid[1]+suction_points[1]]), camera_info))
            '''
            here for each desired suction point coordinate which we saved in the suction_cooridantes is used to find the nearest cooridnate of the suction cup
            This object_centroid[0]+suction_point is same as cooridnates appended in the object_suction_coorindates
            '''
            suction_point = self.find_nearest(point_cloud, [
                                              object_centroid[0]+suction_points[0], object_centroid[1]+suction_points[1]], camera_info)
            cv2.circle(rgb_img, (int(round(suction_point[0])), int(
                round(suction_point[1]))), 1, (255, 0, 0), 5)
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
        y_axis_angle = 0
        '''
        These are the 3D rotation matrix around each axis 
        '''
        R_y = np.array([[math.cos(math.pi*y_axis_angle/180), 0, math.sin(math.pi*y_axis_angle/180)],
                       [0, 1, 0], [-math.sin(math.pi*y_axis_angle/180), 0, math.cos(math.pi*y_axis_angle/180)]])
        '''
        These are the trasnformed cooridnates when the suction cup is rotated at an angle around the obejct centroid
        '''
        for suction in suction_coordinates:
            '''
            These are the base coordinates of suction which are rotated with respect to origin (0,0,0)
            '''
            transformed_base_pose = np.dot(R_y, np.transpose(suction))
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
            transformed_suction_projections = np.vstack((transformed_suction_projections, np.array([point_cloud[0][suction_point[1]][suction_point[0]],
                                                                                                    point_cloud[1][suction_point[1]][suction_point[0]], depth[suction_point[1], suction_point[0]]+transformed_base_pose[2]])))
            # transformed_suction_projections = np.vstack((transformed_suction_projections, R_y*suction))
            # print("appending", transformed_suction_projections)

        print("without transformation", suction_projections)
        print("with transformations", transformed_suction_projections)
        # print(self.convert_pixel_to_point_cloud(depth[265, 290], np.array([265, 290]), camera_info))
        print(object_centroid[0]+0.02, object_centroid[1])

        # print(suction_coordinates)
        # print(object_suction_coordinate)

        cv2.imshow("rgb image", rgb_img)
        cv2.waitKey(0)

        # point_cloud = create_point_cloud_from_depth_image(depth, camera_info)

        # # valid_idx = obj_mask & (point_cloud[..., 2] != 0)
        # point_cloud_valid = seg_mask[:,:,None]*point_cloud
        # x = point_cloud_valid[:,:,0].flatten()
        # y = point_cloud_valid[:,:,1].flatten()
        # z = point_cloud_valid[:,:,2].flatten()
        # print(np.mean(x), np.min(x), np.max(x))
        # print(np.mean(y), np.min(y), np.max(y))
        # print(np.mean(z), np.min(z), np.max(z))

        # centroid = (sum(x) / len(point_cloud_valid), sum(y) / len(point_cloud_valid))
        # print(centroid)
        # print(point_cloud)


if __name__ == "__main__":
    input_image_rgb = cv2.imread(
        "/home/soofiyan_ws/workspaces/aurmr_perception/UIE_main/data_full_pod/Dataset/rgbImages/Kinect/Cropped_RGB/crop_rgb_3H.png")
    input_image_depth = np.load(
        "/home/soofiyan_ws/workspaces/aurmr_perception/UIE_main/data_full_pod/Dataset/depthImages/Kinect/Crop_depth_numpy/crop_depth_3H.npy")

    object = SegnetV2()
    masks, mask_crop = object.mask_generator(input_image_rgb)

    object = run_normal_std()
    object.inference(input_image_rgb, input_image_depth, mask_crop)
