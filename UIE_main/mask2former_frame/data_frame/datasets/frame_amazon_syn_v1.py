import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import cv2
import json
import os
# from ....datasets import transforms as T

from detectron2.structures import (
    BoxMode
)
#
# from utils import mask as util_
# from fcn.config import cfg

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

__all__ = ['frameBinDataset']

data_loading_params = {
    # Camera/Frustum parameters
    'img_width': 256,
    'img_height': 256,
    'near': 0.01,
    'far': 100,
    'fov': 45,  # vertical field of view in degrees

    'use_data_augmentation': True,

    # Multiplicative noise
    'gamma_shape': 1000.,
    'gamma_scale': 0.001,

    # Additive noise
    'gaussian_scale': 0.005,  # 5mm standard dev
    'gp_rescale_factor': 4,
    'gaussian_scale_range': [0., 0.003],
    'gp_rescale_factor_range': [12, 20],

    # Random ellipse dropout
    'ellipse_dropout_mean': 10,
    'ellipse_gamma_shape': 5.0,
    'ellipse_gamma_scale': 1.0,

    # Random high gradient dropout
    'gradient_dropout_left_mean': 15,
    'gradient_dropout_alpha': 2.,
    'gradient_dropout_beta': 5.,

    # Random pixel dropout
    'pixel_dropout_alpha': 1.,
    'pixel_dropout_beta': 10.,
    'crop': True
}

class frameBinDataset(Dataset):
    def __init__(self, file_list, for_training=False, crop=False, num_frame=2):
        # Loads the data from the h5 file, where file is the path to the .h5 file
        self.files = []
        self.lens = []
        # Determines whether to apply data randomizations for training
        self.for_training = for_training
        # Load data from each file
        for i in range(len(file_list)):
            self.files.append(h5py.File(file_list[i], 'r'))
            self.lens.append(self.files[i]['frame0_data'].shape[0])
        self.name = "Bin Dataset"
        self.num_classes = 2
        self.crop = crop
        # self.normalize = T.Compose([
        #     T.ToPiLImage(),
        #     T.ToTensor(),
        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])
        self.num_frame = num_frame

    def __len__(self):
        return sum(self.lens)

    def add_noise_to_depth(self, depth_img, noise_params):
        """ Add noise to depth image.
            This is adapted from the DexNet 2.0 code.
            Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

            @param depth_img: a [H x W] set of depth z values
        """
        depth_img = depth_img.copy()

        # Multiplicative noise: Gamma random variable
        multiplicative_noise = np.random.gamma(noise_params['gamma_shape'], noise_params['gamma_scale'])
        depth_img = multiplicative_noise * depth_img

        return depth_img

    def add_noise_to_xyz(self, xyz_img, depth_img, noise_params):
        """ Add (approximate) Gaussian Process noise to ordered point cloud

            @param xyz_img: a [H x W x 3] ordered point cloud
        """
        xyz_img = xyz_img.copy()

        H, W, C = xyz_img.shape

        # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
        #                 which is rescaled with bicubic interpolation.
        gp_rescale_factor = np.random.randint(noise_params['gp_rescale_factor_range'][0],
                                              noise_params['gp_rescale_factor_range'][1])
        gp_scale = np.random.uniform(noise_params['gaussian_scale_range'][0],
                                     noise_params['gaussian_scale_range'][1])

        small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
        additive_noise = np.random.normal(loc=0.0, scale=gp_scale, size=(small_H, small_W, C))
        additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
        xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]

        return xyz_img

    def dropout_random_ellipses(self, depth_img, noise_params):
        """ Randomly drop a few ellipses in the image for robustness.
            This is adapted from the DexNet 2.0 code.
            Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

            @param depth_img: a [H x W] set of depth z values
        """
        depth_img = depth_img.copy()

        # Sample number of ellipses to dropout
        num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

        # Sample ellipse centers
        nonzero_pixel_indices = np.array(np.where(depth_img > 0)).T  # Shape: [#nonzero_pixels x 2]
        dropout_centers_indices = np.random.choice(nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
        dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :]  # Shape: [num_ellipses_to_dropout x 2]

        # Sample ellipse radii and angles
        x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'],
                                  size=num_ellipses_to_dropout)
        y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'],
                                  size=num_ellipses_to_dropout)
        angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

        # Dropout ellipses
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            # dropout the ellipse
            mask = np.zeros_like(depth_img)
            mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360,
                               color=1, thickness=-1)
            depth_img[mask == 1] = 0

        return depth_img

    def random_color_warp(self, image, d_h=None, d_s=None, d_l=None):
        """ Given an RGB image [H x W x 3], add random hue, saturation and luminosity to the image

            Code adapted from: https://github.com/yuxng/PoseCNN/blob/master/lib/utils/blob.py
        """
        H, W, _ = image.shape

        image_color_warped = np.zeros_like(image)

        # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
        if d_h is None:
            d_h = (np.random.uniform() - 0.5) * 0.2 * 256
        if d_l is None:
            d_l = (np.random.uniform() - 0.5) * 0.2 * 256
        if d_s is None:
            d_s = (np.random.uniform() - 0.5) * 0.2 * 256

        # Convert the RGB to HLS
        hls = cv2.cvtColor(image.round().astype(np.uint8), cv2.COLOR_RGB2HLS)
        h, l, s = cv2.split(hls)

        # Add the values to the image H, L, S
        # new_h = (np.round((h + d_h)) % 256).astype(np.uint8)
        new_h = np.round(np.clip(h + d_h, 0, 255)).astype(np.uint8)
        new_l = np.round(np.clip(l + d_l, 0, 255)).astype(np.uint8)
        new_s = np.round(np.clip(s + d_s, 0, 255)).astype(np.uint8)

        # Convert the HLS to RGB
        new_hls = cv2.merge((new_h, new_l, new_s)).astype(np.uint8)
        new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2RGB)

        image_color_warped = new_im.astype(np.float32)

        return image_color_warped

    # Computes point cloud from depth image and camera intrinsics
    def compute_xyz(self, depth_img, intrinsic):
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        px = intrinsic[0][2]
        py = intrinsic[1][2]
        height, width = depth_img.shape

        indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
        z_e = depth_img
        x_e = (indices[..., 1] - px) * z_e / fx
        y_e = (indices[..., 0] - py) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
        return xyz_img

    def random_crop(self, img, mask, depth):
        # Randomly choose a valid height and width
        height = np.random.randint(0, img.shape[0] * .4)
        width = np.random.randint(0, img.shape[1] * .4)
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        x = np.random.randint(0, img.shape[1] - width)
        y = np.random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, ...]
        mask = mask[y:y + height, x:x + width]
        depth = depth[y:y + height, x:x + width, ...]
        return img, mask, depth

    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels

    def pad_crop_resize(self, img, label, depth):
        """ Crop the image around the label mask, then resize to 224x224
        """

        H, W, _ = img.shape

        # sample an object to crop
        K = np.max(label)
        while True:
            if K > 0:
                idx = np.random.randint(1, K + 1)
            else:
                idx = 0
            foreground = (label == idx).astype(np.float32)

            # get tight box around label/morphed label
            if np.sum(foreground) == 0:
                continue

            x_min, y_min, x_max, y_max = util_.mask_to_tight_box(foreground)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            # make bbox square
            x_delta = x_max - x_min
            y_delta = y_max - y_min
            if x_delta > y_delta:
                y_min = cy - x_delta / 2
                y_max = cy + x_delta / 2
            else:
                x_min = cx - y_delta / 2
                x_max = cx + y_delta / 2

            sidelength = x_max - x_min
            padding_percentage = np.random.uniform(cfg.TRAIN.min_padding_percentage, cfg.TRAIN.max_padding_percentage)
            padding = int(round(sidelength * padding_percentage))
            if padding == 0:
                padding = 25

            # Pad and be careful of boundaries
            x_min = max(int(x_min - padding), 0)
            x_max = min(int(x_max + padding), W - 1)
            y_min = max(int(y_min - padding), 0)
            y_max = min(int(y_max + padding), H - 1)

            # crop
            if (y_min == y_max) or (x_min == x_max):
                continue

            img_crop = img[y_min:y_max + 1, x_min:x_max + 1]
            label_crop = label[y_min:y_max + 1, x_min:x_max + 1]
            roi = [x_min, y_min, x_max, y_max]
            if depth is not None:
                depth_crop = depth[y_min:y_max + 1, x_min:x_max + 1]
            break

        # resize
        s = cfg.TRAIN.SYN_CROP_SIZE
        img_crop = cv2.resize(img_crop, (s, s))
        label_crop = cv2.resize(label_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        if depth is not None:
            depth_crop = cv2.resize(depth_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        else:
            depth_crop = None

        return img_crop, label_crop, depth_crop

    @staticmethod
    def mask2bbox(mask):
        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0

        return np.array([x1, y1, x2, y2])

    def get_frame_from_image(self, image_index, frame_id, color_jitter_param, flip):
        """
        Returns:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
                'image': HxWxC, uint8, RGB
                'xyz': HxWx3
                'height', 'width'
                'image_id':
                'annotation': list of dict
                    'iscrowd': 0
                    'bbox': [x1, y1, x2, y2]
                    'category_id']: 1
                    'segmentation': binary_mask
                    'bbox_mode': BoxMode.XYXY_ABS
        """
        # For now, we only care about the more packed frame
        cat = f'{frame_id}_'

        # Find the right file to consider and the matching image_index
        file_now = 0
        while image_index >= self.lens[file_now]:
            image_index -= self.lens[file_now]
            file_now += 1
        self.file = self.files[file_now]

        # Loads the color image
        rgb = self.file[cat + 'data'][image_index][..., 0:3]
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Applies color jittering
        if self.for_training:
            rgb = self.random_color_warp(rgb, **color_jitter_param)

        # Loads the depth image and camera intrinsics
        depth = self.file[cat + 'depth'][image_index]

        # Finds all valid pixels for standardization
        mask = depth >= -1000
        depth[depth < -1000] = 0
        intrinsic = json.loads(self.file[cat + 'metadata'][image_index])['camera']['intrinsic_matrix']

        # Adds noise and cuts out random ellipses from depth
        if self.for_training:
            depth = self.add_noise_to_depth(depth, data_loading_params)

        # Computes the point cloud
        xyz = self.compute_xyz(depth, intrinsic).astype(np.float32)

        # Adds noise to the point cloud
        if self.for_training:
            xyz = self.add_noise_to_xyz(xyz, depth, data_loading_params)

        # Loads the label masks
        label = self.file[cat + 'mask'][image_index]
        label = (label + 1) % 65536
        label %= np.max(label)

        # Randomly flip the image/point cloud/label
        if self.for_training:
            if flip:
                rgb = np.fliplr(rgb).copy()
                depth = np.fliplr(depth).copy()
                xyz = np.fliplr(xyz).copy()
                label = np.fliplr(label).copy()

        if self.crop:
            rgb, label, xyz = self.pad_crop_resize(rgb, label, xyz)
            label = self.process_label(label)
            # mask = xyz[...,2] > .05

        # Adds noise and cuts out random ellipses from depth
        if self.for_training:
            xyz[..., 2] = self.dropout_random_ellipses(xyz[..., 2], data_loading_params)
            xyz[..., 0][xyz[..., 2] == 0] = 0
            xyz[..., 1][xyz[..., 2] == 0] = 0

        # Standardize image and point cloud
        # mask = xyz[..., 2] > .05
        # xyz[..., 0] = (xyz[..., 0] - np.mean(xyz[..., 0], where=mask)) / np.std(xyz[..., 0], where=mask) * mask
        # xyz[..., 1] = (xyz[..., 1] - np.mean(xyz[..., 1], where=mask)) / np.std(xyz[..., 1], where=mask) * mask
        # xyz[..., 2] = (xyz[..., 2] - np.mean(xyz[..., 2], where=mask)) / np.std(xyz[..., 2], where=mask) * mask

        annotations = []
        for label_idx in np.unique(label):
            if label_idx==0:
                continue
            obj_anno = {'iscrowd': 0,
                        'bbox': self.mask2bbox(label==label_idx),
                        'category_id': 0,
                        'segmentation': label==label_idx,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'track_id': label_idx,
                        }
            annotations.append(obj_anno)
        sample = {'image': rgb,
                  'xyz': xyz,
                  'height': rgb.shape[0],
                  'width': rgb.shape[1],
                  'image_id': image_index,
                  'annotations': annotations
                  }

        return sample

    def __getitem__(self, idx):
        if self.for_training:
            d_h = (np.random.uniform() - 0.5) * 0.2 * 256
            d_l = (np.random.uniform() - 0.5) * 0.2 * 256
            d_s = (np.random.uniform() - 0.5) * 0.2 * 256
            color_jitter_param = {'d_h': d_h, 'd_l': d_l, 'd_s': d_s}
            flip = np.random.uniform() < .5
        else:
            color_jitter_param = {'d_h': 0, 'd_l': 0, 'd_s': 0}
            flip = False
        data_0 = self.get_frame_from_image(idx, 'frame0', color_jitter_param, flip)
        data_1 = self.get_frame_from_image(idx, 'frame1', color_jitter_param, flip)
        return [data_0, data_1]
    
    def load_dataset_into_memory(self):
        files = []
        lens = []
        for h5_file in self.files:
            files.append({k: h5_file[k][:] for k in h5_file.keys()})
            lens.append(len(files[-1]['frame0_data']))
        self.files = files
        self.lens = lens
        

def build_amazon_syn_v1(image_set, args):
    if image_set == 'train':
        file_list = [os.path.join(args.amazon_path_train, x) for x in args.train_split]
    elif image_set == 'val':
        file_list = [os.path.join(args.amazon_path_val, x) for x in args.val_split]
    return frameBinDataset(file_list, for_training=image_set=='train')

if __name__ == "__main__":
    file_name = '/home/thomas/Desktop/azure_data/azure_24_128.npy'
    data = np.load(file_name, allow_pickle=True).item()
    dataset = frameBinDataset(data)
    breakpoint()