'''
MIT License

Copyright (c) 2022 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import json
import numpy as np
import re
import torch
import torch.nn.functional as F


class NormalizeInverse(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

denormalize_rgb = transforms.Compose([
    NormalizeInverse(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
    transforms.ToPILImage(),
])

normalize_depth = transforms.Normalize(
    mean=0.9915893,
    std=0.31949179
)

denormalize_depth = NormalizeInverse(
    mean=0.9915893,
    std=0.31949179
)

normalize_xyz = transforms.Normalize(
    mean=[-0.2605608, -1.2007151, 1.6376148],
    std=[1.87574718, 1.87346588, 1.95350732]
)

denormalize_xyz = NormalizeInverse(
    mean=[-0.2605608, -1.2007151, 1.6376148],
    std=[1.87574718, 1.87346588, 1.95350732]
)

normalize_xyz_world = transforms.Normalize(
    mean=[-0.7235448, -3.3750634, -0.42362145],
    std=[2.38134491, 2.58399743, 1.30816852]
)

denormalize_xyz_world = NormalizeInverse(
    mean=[-0.7235448, -3.3750634, -0.42362145],
    std=[2.38134491, 2.58399743, 1.30816852]
)

def build_predicates(objects, unary, binary):
    pred_names = [pred % obj for pred in unary for obj in objects]
    obj1 = [obj for _ in range(len(objects) - 1) for obj in objects]
    obj2 = [obj for i in range(1, len(objects)) for obj in np.roll(objects, -i)]
    pred_names += [pred % (o1, o2) for pred in binary for o1, o2 in zip(obj1, obj2)]
    return pred_names


def image_transform(crop, scale):
    rot = np.eye(3)
    rot[:2, :2] = np.diag(scale)
    trans = np.eye(3)
    trans[:2, 2] = -np.array(crop[:2])
    return torch.from_numpy(rot @ trans).float()


def tokenize(symbol):
    """ Split a predicate name or action """
    tks = re.split('([,-.;" (){}])', symbol)
    tks = [tk.strip() for tk in tks]
    tks = [tk.lower().replace("“","").replace("”","") for tk in tks if len(tk) > 0]
    tks = [tk.strip('(').strip(')').strip(',') for tk in tks]
    tks = [tk for tk in tks if len(tk) > 0]
    return tks


def action_to_language(action):
    verb, with_obj, to_obj = tokenize(action)
    if verb == "grasp":
        w, t = None, to_obj
        opts = ["grab %s", "grasp %s", "take %s"]
    elif verb == "approach":
        opts = ["reach for %s", "approach %s", "line gripper up with %s",
                "get ready to take %s", "align with %s"]
        w, t = None, to_obj
    elif verb == "stack":
        w, t = with_obj, to_obj
        opts = ["stack %s on top of %s", "put %s on %s", "place %s on %s", "stack %s on %s"]
    elif verb == "align":
        w, t = with_obj, to_obj
        opts = ["line %s up with %s", "move %s over top of %s", "align %s with %s",
                "move %s over %s"]
    elif verb == "place":
        t, w = with_obj, to_obj
        opts = ["put %s on %s", "move %s onto %s", "drop %s on %s",
                "place %s on %s"]
    elif verb == "lift":
        w, t = None, to_obj
        opts = ["lift %s", "pick up %s", "raise %s"]
    elif verb == "release":
        w, t = None, to_obj
        opts = ["release %s", "let go of %s", "open gripper with %s"]

    i = np.random.randint(len(opts))
    action = opts[i]
    if w is None:
        return action % t
    else:
        return action % (w, t)


class AURMRDataset(Dataset):
    def __init__(self, file_path, patch_size, max_nobj):
        self.file_path = file_path
        self.h5 = h5py.File(self.file_path)
        self.patch_size = patch_size
        self.max_nobj = max_nobj

    def __len__(self):
        return self.h5['frame0_data'].shape[0]
    
    def get_obj_masks(self, md, mask, add_empty_target=False):
        masks = []
        to_tensor = transforms.ToTensor()
        num_objs = len(md['name_to_id_map'].keys()) - 1
        for i in range(num_objs):
            obj_mask = mask == i
            tensor = to_tensor(Image.fromarray(obj_mask).resize((self.patch_size, self.patch_size)))
            masks.append(tensor.squeeze())
        if add_empty_target:
            masks.append(torch.zeros((self.patch_size, self.patch_size)))
        for _ in range(len(masks), self.max_nobj):
            masks.append(torch.zeros((self.patch_size, self.patch_size)) - 1)
        return torch.stack(masks)

    def __getitem__(self, idx):
        md0 = json.loads(self.h5['frame0_metadata'][idx])
        md1 = json.loads(self.h5['frame1_metadata'][idx])
        mask0 = self.h5['frame0_mask'][idx]
        mask1 = self.h5['frame1_mask'][idx]

        input_obj_masks = self.get_obj_masks(md0, mask0, add_empty_target=True)
        label_obj_masks = self.get_obj_masks(md1, mask1)

        output = {
            'rgb0': normalize_rgb(Image.fromarray(self.h5['frame0_data'][idx]).convert('RGB')),
            'rgb1': normalize_rgb(Image.fromarray(self.h5['frame1_data'][idx]).convert('RGB')),
            'depth0': torch.from_numpy(self.h5['frame0_depth'][idx]),
            'depth1': torch.from_numpy(self.h5['frame1_depth'][idx]),
            'input_masks': input_obj_masks,
            'label_masks': label_obj_masks,
        }
        
        return output


class CLEVRDataset(Dataset):
    def __init__(self, scene_file, obj_file, max_nobj, rand_patch):
        self.obj_file = obj_file
        self.obj_h5 = None
        self.scene_file = scene_file
        self.scene_h5 = None
        with h5py.File(scene_file) as scene_h5:
            self.scenes = list(scene_h5.keys())
        self.max_nobj = max_nobj
        self.rand_patch = rand_patch

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        if self.obj_h5 is None:
            self.obj_h5 = h5py.File(self.obj_file)
        if self.scene_h5 is None:
            self.scene_h5 = h5py.File(self.scene_file)

        scene = self.scene_h5[self.scenes[idx]]
        img = normalize_rgb(Image.open(BytesIO(scene['image'][()])).convert('RGB'))

        objects = scene['objects'][()].decode().split(',')
        obj_patches = []
        for obj in objects:
            patch_idx = 0
            if self.rand_patch:
                patch_idx = torch.randint(len(self.obj_h5[obj]), ()).item()
            patch = normalize_rgb(Image.open(BytesIO(self.obj_h5[obj][patch_idx])))
            obj_patches.append(patch)
        for _ in range(len(obj_patches), self.max_nobj):
            obj_patches.append(torch.zeros_like(obj_patches[0]))
        obj_patches = torch.stack(obj_patches)

        relations, mask = [], []
        ids = np.arange(self.max_nobj)
        for relation in scene['relations']:
            for k in range(1, self.max_nobj):
                for i, j in zip(ids, np.roll(ids, -k)):
                    if i >= len(objects) or j >= len(objects):
                        relations.append(0)
                        mask.append(0)
                    else:
                        relations.append(relation[i][j])
                        mask.append(relation[i][j] != -1)
        relations = torch.tensor(relations).float()
        mask = torch.tensor(mask).float()

        return img, obj_patches, relations, mask


class LeonardoDataset(Dataset):
    def __init__(
        self, data_type, data_dir, split, obj_file, n_objects, depth=False,
        xyz=True, world=True, obj_rgb=True, obj_depth=False, randpatch=True,
        supervisions=['cls'], predicates=None, view=1, multiview=True,
        randview=True, min_crop_ratio=1., visible_thresh=10,
        normalize_rgb=True, normalize_target=False, img_size=(224, 224),
        patch_size=(32, 32), focal=221.7025
    ):
        with open(f'{data_dir}/{split}_nframes.json') as f:
            n_frames = json.load(f)
            self.sequences = list(n_frames.keys())
            n_frames = list(n_frames.values())
            self.cum_n_frames = np.cumsum(n_frames)

        self.predicates = predicates
        self.pred_ids = None
        self.object_seg_ids = None
        with h5py.File(f'{data_dir}/{split}.h5') as data:
            if 'predicates' in data:
                predicates = data['predicates'][()].decode().split('|')
                self.pred_ids = {pred: i for i, pred in enumerate(predicates)}
            if 'object_seg_ids' in data:
                self.object_seg_ids = data['object_seg_ids'][()]

        self.data_type = data_type
        if data_type == 'leonardo':
            self.preconditions = json.load(
                open(f'preconditions/{data_type}_{n_objects}obj.json')
            )

        self.data_dir = data_dir
        self.split = split
        self.h5 = None
        self.supervisions = supervisions

        self.obj_file = obj_file
        self.n_objects = n_objects
        self.obj_h5 = None
        self.obj_rgb = obj_rgb
        self.obj_depth = obj_depth
        self.randpatch = randpatch

        self.view = view
        self.multiview = multiview
        self.randview = randview

        self.depth = depth
        self.xyz = xyz
        self.world = world
        self.focal_length = focal

        self.min_crop_ratio = min_crop_ratio
        self.visible_thresh = visible_thresh
        self.normalize_rgb = normalize_rgb
        self.normalize_target = normalize_target
        self.img_size = img_size
        self.patch_size = patch_size

    def __len__(self):
        return self.cum_n_frames[-1]

    def load_h5(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(f'{self.data_dir}/{self.split}.h5', 'r')
        if self.obj_h5 is None:
            self.obj_h5 = h5py.File(self.obj_file, 'r')
            self.obj_ids = {
                obj: i for i, obj in enumerate(sorted(self.obj_h5.keys()))
            }
        # Get H5 file index and frame index
        file_idx = np.argmax(idx < self.cum_n_frames)
        data = self.h5[self.sequences[file_idx]]
        frame_idx = idx
        if file_idx > 0:
            frame_idx = idx - self.cum_n_frames[file_idx - 1]
        return data, frame_idx

    def get_depth(self, data, idx, v):
        depth = Image.open(BytesIO(data[f'depth{v}'][idx]))
        dmin = data[f'depth_min{v}'][idx]
        dmax = data[f'depth_max{v}'][idx]
        depth = np.array(depth).astype(np.uint16)
        depth = depth / 1000. * (dmax - dmin) + dmin
        depth[depth < 0.01] = dmax + np.random.uniform(
            0.2, 0.5, (depth < 0.01).sum()
        )
        return depth

    def depth2xyz(self, depth, img_size, cam):
        H, W = img_size
        x, y = np.meshgrid(range(W), range(H))
        xyz = np.stack([x * depth, y * depth, depth]).reshape(3, -1)
        K = np.eye(3)
        K[0, 0] = K[1, 1] = self.focal_length
        K[0, 2] = (W - 1) / 2
        K[1, 2] = (H - 1) / 2
        xyz = np.linalg.inv(K) @ xyz
        if self.world:
            xyz = np.concatenate([xyz, np.ones((1, xyz.shape[1]))])
            xyz = np.linalg.inv(cam) @ xyz
        xyz = xyz[:3].reshape(3, H, W)
        return xyz

    def get_imgs(self, data, idx, v):
        rgb = Image.open(BytesIO(data[f'rgb{v}'][idx])).convert('RGB')
        seg = np.array(Image.open(BytesIO(data[f'seg{v}'][idx])))
        cam = np.squeeze(data[f'view_matrix{v}'][()])
        depth = None
        if self.depth:
            depth = self.get_depth(data, idx, v)
            if self.xyz:
                depth = self.depth2xyz(depth, depth.shape, cam)
            else:
                depth = depth[None, ...]
            depth = torch.from_numpy(depth).float()
        crop_tr = torch.eye(3).float()
        if self.min_crop_ratio < 1:
            i, j, h, w = transforms.RandomResizedCrop(self.img_size).get_params(
                rgb, (self.min_crop_ratio, 1.0), (1.0, 1.0)
            )
            crop_tr = image_transform(
                [j, i], [self.img_size[0] / w, self.img_size[1] / h]
            )
            rgb = transforms.functional.crop(rgb, i, j, h, w)
            seg = seg[i:i+h, j:j+w]
            if self.depth:
                depth = depth[:, i:i+h, j:j+w]
        if self.normalize_rgb:
            rgb = normalize_rgb(rgb.resize(self.img_size))
        else:
            rgb = transforms.functional.to_tensor(rgb.resize(self.img_size))
        if self.depth:
            depth = transforms.functional.resize(depth, self.img_size)
            if self.xyz:
                if self.world:
                    depth = normalize_xyz_world(depth)
                else:
                    depth = normalize_xyz(depth)
            else:
                depth = normalize_depth(depth)
        cam = torch.from_numpy(cam).float()
        return rgb, depth, seg, cam, crop_tr

    def get_objects(self, data):
        objects = data['colors'][()].decode().split(',')[:self.n_objects]
        obj_names = [f'{color}_block' for color in objects]
        canon_objects = [f'object{i:02d}' for i in range(len(objects))]
        return objects, obj_names, canon_objects

    def get_patches(self, colors):
        obj_patches = []
        for color in colors:
            patch_idx = 0
            if self.randpatch:
                patch_idx = torch.randint(len(self.obj_h5[color]), ()).item()
            patch = Image.open(BytesIO(self.obj_h5[color][patch_idx]))
            obj_patches.append(normalize_rgb(patch.resize(self.patch_size)))
        return torch.stack(obj_patches), None

    def get_seg_ids(self, data, objects):
        seg_ids = self.object_seg_ids
        if seg_ids is None:
            seg_ids = [
                data['segmentation_id'][obj][()] for obj in objects
            ]
        return seg_ids

    def get_visible(self, segs, seg_ids):
        pix_count = [sum([(seg == k).sum() for seg in segs]) for k in seg_ids]
        visible = torch.zeros(self.n_objects).bool()
        visible[:len(seg_ids)] = torch.tensor(pix_count) >= self.visible_thresh
        return visible

    def get_joint(self, data, idx):
        joint = data['joint'][idx].astype('float32') / np.pi
        gripper = data['gripper'][idx:idx+1].astype('float32') / 0.04 - 0.5
        joint = np.concatenate([joint, gripper])
        return torch.from_numpy(joint)

    def get_predicates(self, data, idx, pred_ids):
        predicates = torch.from_numpy(
            data['logical'][idx][[pred_ids[pred] for pred in self.predicates]]
        ).float()
        return predicates

    def get_preconditions(self, data, idx):
        plan = data['sym_plan'][()].decode().split(',')
        mask = torch.zeros(len(self.predicates)).float()
        if idx >= len(plan):
            action = None
        else:
            action = plan[idx]
            pred_ids = {pred: i for i, pred in enumerate(self.predicates)}
            for cond, val in self.preconditions[action]:
                mask[pred_ids[cond]] = 1
        # print(action)
        # for pred, m in zip(self.predicates, mask):
        #     if m:
        #         print(pred)
        # exit(0)
        return action, mask

    def get_ee_xyz(self, data, idx):
        return torch.from_numpy(data['ee_pose'][idx][:3, 3]).float()

    def get_obj_xyz(self, data, idx, objects):
        obj_xyz = torch.stack([
            torch.from_numpy(data[f'{obj}_pose'][idx][:3, 3]).float()
            for obj in objects
        ])
        return obj_xyz

    def get_ee_obj_xyz(self, data, idx, objects):
        xyz = torch.stack([torch.from_numpy(
            data[f'{obj}_pose'][idx][:3, 3] - data['ee_pose'][idx][:3, 3]
        ) for obj in objects]).float()
        return xyz

    def get_obj_obj_xyz(self, data, idx, objects):
        obj1 = [obj for _ in range(len(objects) - 1) for obj in objects]
        obj2 = [obj for i in range(1, len(objects)) for obj in np.roll(objects, -i)]
        xyz = torch.stack([torch.from_numpy(
            data[f'{o2}_pose'][idx][:3, 3] - data[f'{o1}_pose'][idx][:3, 3]
        ) for o1, o2 in zip(obj1, obj2)]).float()
        return xyz

    def get_ee_obj_xyz_new(self, data, idx, objects, logical, pred_ids):
        obj_in_hand = None
        targets = []
        for obj in objects:
            if logical[pred_ids[f'has_obj(robot, {obj})']]:
                obj_in_hand = obj
            targets.append(data[f'{obj}_pose'][idx][:3, 3])
        if obj_in_hand is not None:
            for i, obj in enumerate(objects):
                if obj == obj_in_hand:
                    targets[i] = data['ee_pose'][idx][:3, 3]
                else:
                    targets[i][2] += 0.05
        xyz = torch.stack([
            torch.from_numpy(target - data['ee_pose'][idx][:3, 3])
            for target in targets
        ]).float()
        if self.normalize_target:
            norm = xyz.norm(dim=1, keepdims=True)
            xyz = torch.cat([xyz / norm, norm], dim=1)
        return xyz

    def get_item(self, data, idx):
        output = {}

        # Load images from H5 file
        if self.multiview:
            rgbs, depths, segs, cams, crop_trs = zip(*[
                self.get_imgs(data, idx, v) for v in range(self.view)
            ])
            rgb = torch.cat(rgbs, dim=1)
            depth = torch.cat(depths, dim=1) if self.depth else torch.empty(())
            cam = torch.stack(cams)
            crop_tr = torch.stack(crop_trs)
        else:
            v = torch.randint(self.view, ()).item() if self.randview else self.view
            rgb, depth, seg, cam, crop_tr = self.get_imgs(data, idx, v)
            depth = torch.empty() if depth is None else depth
            segs = [seg]
        output['rgb'] = rgb
        output['depth'] = depth
        output['cam'] = cam
        output['crop_tr'] = crop_tr

        # Load object patches
        objects, obj_names, canon_objects = self.get_objects(data)
        output['obj_names'] = obj_names
        if self.obj_rgb:
            output['obj_patches'], obj_depth = self.get_patches(objects)
            if self.obj_depth:
                output['obj_depth'] = obj_depth

        # Load mask for visible objects
        output['obj_xyz'] = self.get_obj_xyz(data, idx, canon_objects)
        output['visible'] = self.get_visible(
            segs, self.get_seg_ids(data, canon_objects)
        )
        output['visible'][output['obj_xyz'].max(dim=1)[0] > 10] = 0

        # Load gripper state from H5 file
        output['joint'] = self.get_joint(data, idx)

        # Load predicates from H5 file
        if 'cls' in self.supervisions or 'reg_new' in self.supervisions:
            if 'predicate_id' in data:
                pred_ids = {
                    pred: data['predicate_id'][pred][()]
                    for pred in data['predicate_id']
                }
            else:
                pred_ids = self.pred_ids

        if 'cls' in self.supervisions:
            output['predicates'] = self.get_predicates(data, idx, pred_ids)
            if self.data_type == 'leonardo':
                output['action'], output['precond_mask'] = \
                    self.get_preconditions(data, idx)

        # Load regression targets
        if 'reg' in self.supervisions or 'reg_new' in self.supervisions:
            output['ee_xyz'] = self.get_ee_xyz(data, idx)
        if 'reg' in self.supervisions:
            ee_obj_xyz = self.get_ee_obj_xyz(data, idx, canon_objects)
            obj_obj_xyz = self.get_obj_obj_xyz(data, idx, canon_objects)
            output['targets'] = torch.cat(
                [ee_obj_xyz.T.reshape(-1), obj_obj_xyz.T.reshape(-1)]
            )
        elif 'reg_new' in self.supervisions:
            ee_obj_xyz = self.get_ee_obj_xyz_new(
                data, idx, canon_objects, data['logical'][idx], pred_ids
            )
            output['targets'] = ee_obj_xyz.T.reshape(-1)

        return output

    def __getitem__(self, idx):
        data, frame_idx = self.load_h5(idx)
        output = self.get_item(data, frame_idx)
        return output


class KitchenDataset(LeonardoDataset):
    def get_objects(self, data):
        objects = data['objects'][()].decode().split(',')[:self.n_objects]
        obj_names = [
            '_'.join([f'{i:02d}'] + obj.split('_')[:-2])
            for i, obj in enumerate(objects)
        ]
        canon_objects = [f'object_{i:02d}' for i in range(len(objects))]
        return objects, obj_names, canon_objects

    def get_patches(self, objects):
        obj_patches = []
        obj_depth = []
        for obj in objects:
            patch_idx = 0
            if self.randpatch:
                patch_idx = torch.randint(len(self.obj_h5[obj]['rgb']), ()).item()
            patch = Image.open(BytesIO(self.obj_h5[obj]['rgb'][patch_idx]))
            patch = normalize_rgb(patch.resize(self.patch_size))
            obj_patches.append(patch)
            if self.obj_depth:
                depth = self.get_depth(self.obj_h5[obj], patch_idx, '')
                if self.xyz:
                    depth = self.depth2xyz(depth, depth.shape)
                else:
                    depth = depth[None, ...]
                depth = torch.from_numpy(depth).float()
                depth = transforms.functional.resize(depth, self.patch_size)
                if self.xyz:
                    depth = normalize_xyz(depth)
                else:
                    depth = normalize_depth(depth)
                obj_depth.append(depth)

        obj_patches = F.pad(
            torch.stack(obj_patches),
            (0, 0, 0, 0, 0, 0, 0, max(self.n_objects - len(objects), 0))
        )
        if self.obj_depth:
            obj_depth = F.pad(
                torch.stack(obj_depth),
                (0, 0, 0, 0, 0, 0, 0, max(self.n_objects - len(objects), 0))
            )
        else:
            obj_depth = torch.empty(())

        return obj_patches, obj_depth

    def get_predicates(self, data, idx, pred_ids):
        # Load predicates from H5 file
        predicates = []
        for pred in self.predicates:
            if pred in pred_ids:
                pred_ids[pred] = data['predicate_id'][pred][()]
                predicates.append(data['logical'][idx][pred_ids[pred]])
            else:
                predicates.append(0)
        return torch.tensor(predicates).float()

    def get_ee_xyz(self, data, idx):
        return torch.from_numpy(data['pose']['ee'][idx][:3, 3]).float()

    def get_obj_xyz(self, data, idx, objects):
        obj_xyz = torch.stack([
            torch.from_numpy(data['pose'][obj][idx][:3, 3]).float()
            for obj in objects
        ])
        obj_xyz = F.pad(obj_xyz, (0, 0, 0, max(self.n_objects - len(objects), 0)))
        return obj_xyz

    def get_ee_obj_xyz(self, data, idx, objects):
        xyz = torch.stack([torch.from_numpy(
            data['pose'][obj][idx][:3, 3] - data['pose']['ee'][idx][:3, 3]
        ) for obj in objects]).float()
        xyz = F.pad(xyz, (0, 0, 0, max(self.n_objects - len(objects), 0)))
        return xyz

    def get_obj_obj_xyz(self, data, idx, objects):
        xyz = []
        for k in range(1, self.n_objects):
            for i, j in zip(
                range(self.n_objects),
                np.roll(np.arange(self.n_objects), -k)
            ):
                if i < len(objects) and j < len(objects):
                    xyz.append(torch.from_numpy(
                        data['pose'][objects[j]][idx][:3, 3]
                        - data['pose'][objects[i]][idx][:3, 3]
                    ).float())
                else:
                    xyz.append(torch.zeros(3).float())
        return torch.stack(xyz)

    def get_ee_obj_xyz_new(self, data, idx, objects, logical, pred_ids):
        obj_in_hand = None
        targets = []
        for obj in objects:
            if logical[pred_ids[f'has_obj(robot, {obj})']]:
                obj_in_hand = obj
            targets.append(data['pose'][obj][idx][:3, 3])
        ee_xyz = data['pose']['ee'][idx][:3, 3]
        if obj_in_hand is not None:
            for i, obj in enumerate(objects):
                if obj == obj_in_hand:
                    targets[i] = data['pose']['ee'][idx][:3, 3]
                else:
                    h1 = data['bbox'][obj][idx][:, 2].max() \
                       - data['bbox'][obj][idx][:, 2].min()
                    bottom2 = data['bbox'][obj_in_hand][idx][:, 2].min()
                    offset = h1 / 2 + ee_xyz[2] - bottom2
                    targets[i][2] += offset
        xyz = torch.stack([
            torch.from_numpy(target - ee_xyz) for target in targets
        ]).float()
        if self.normalize_target:
            norm = xyz.norm(dim=1, keepdims=True)
            xyz = torch.cat([xyz / norm, norm], dim=1)
        xyz = F.pad(xyz, (0, 0, 0, max(self.n_objects - len(objects), 0)))
        return xyz


class LeonardoImageDataset(LeonardoDataset):
    def __init__(
        self, data_dir, split, obj_file, nobj_seq, nobj_data, supervisions,
        predicates=None, unary=None, binary=None,  min_crop_ratio=1.,
        visible_thresh=10, normalize_rgb=True, img_size=(224, 224)
    ):
        super(LeonardoImageDataset, self).__init__(
            'leonardo', data_dir, split, obj_file, nobj_seq, obj_rgb=False,
            supervisions=supervisions, predicates=predicates,
            min_crop_ratio=min_crop_ratio, visible_thresh=visible_thresh,
            normalize_rgb=normalize_rgb, img_size=img_size
        )
        self.nobj_data = nobj_data
        if nobj_data is not None:
            all_objects = [f'object{i:02d}' for i in range(nobj_data)]
            if 'cls' in supervisions:
                all_predicates = build_predicates(all_objects, unary, binary)
                self.all_pred_ids = {
                    pred: i for i, pred in enumerate(all_predicates)
                }
                self.unary = unary
                self.binary = binary
            if 'reg' in supervisions:
                all_targets = build_predicates(
                    all_objects, ['x(ee, %s)', 'y(ee, %s)', 'z(ee, %s)'],
                    ['x(%s, %s)', 'y(%s, %s)', 'z(%s, %s)']
                )
                self.all_target_ids = {
                    target: i for i, target in enumerate(all_targets)
                }

    def __getitem__(self, idx):
        data, frame_idx = self.load_h5(idx)
        output = self.get_item(data, frame_idx)

        if self.nobj_data is not None:
            # Load object IDs
            objects = data['colors'][()].decode().split(',')[:self.n_objects]
            obj_ids = [self.obj_ids[color] for color in objects]
            canon_objects = [f'object{i:02d}' for i in obj_ids]

            # Mapping from model prediction to ground truth
            if 'cls' in self.supervisions:
                predicates = build_predicates(
                    canon_objects, self.unary, self.binary
                )
                output['pred2gt_cls'] = torch.tensor([
                    self.all_pred_ids[pred] for pred in predicates
                ]).long()
            if 'reg' in self.supervisions:
                targets = build_predicates(
                    canon_objects, ['x(ee, %s)', 'y(ee, %s)', 'z(ee, %s)'],
                    ['x(%s, %s)', 'y(%s, %s)', 'z(%s, %s)']
                )
                output['pred2gt_reg'] = torch.tensor([
                    self.all_target_ids[target] for target in targets
                ]).long()

        return output


class KitchenImageDataset(KitchenDataset):
    def __init__(
        self, data_dir, split, obj_file, nobj_seq, nobj_data, supervisions,
        predicates=None, unary=None, binary=None,  min_crop_ratio=1.,
        visible_thresh=10, normalize_rgb=True, img_size=(224, 224)
    ):
        super(KitchenImageDataset, self).__init__(
            'kitchen', data_dir, split, obj_file, nobj_seq, obj_rgb=False,
            supervisions=supervisions, predicates=predicates,
            min_crop_ratio=min_crop_ratio, visible_thresh=visible_thresh,
            normalize_rgb=normalize_rgb, img_size=img_size
        )
        self.nobj_data = nobj_data
        if nobj_data is not None:
            all_objects = [f'object_{i:02d}' for i in range(nobj_data)]
            if 'cls' in supervisions:
                all_predicates = build_predicates(all_objects, unary, binary)
                self.all_pred_ids = {
                    pred: i for i, pred in enumerate(all_predicates)
                }
                self.unary = unary
                self.binary = binary
            if 'reg' in supervisions:
                all_targets = build_predicates(
                    all_objects, ['x(ee, %s)', 'y(ee, %s)', 'z(ee, %s)'],
                    ['x(%s, %s)', 'y(%s, %s)', 'z(%s, %s)']
                )
                self.all_target_ids = {
                    target: i for i, target in enumerate(all_targets)
                }

    def __getitem__(self, idx):
        data, frame_idx = self.load_h5(idx)
        output = self.get_item(data, frame_idx)

        if self.nobj_data is not None:
            # Load object IDs
            objects = data['objects'][()].decode().split(',')[:self.n_objects]
            obj_ids = [self.obj_ids[obj] for obj in objects]
            i = 0
            for _ in range(len(obj_ids), self.n_objects):
                while i in obj_ids:
                    i += 1
                obj_ids.append(i)
            canon_objects = [f'object_{i:02d}' for i in obj_ids]

            # Mapping from model prediction to ground truth
            if 'cls' in self.supervisions:
                predicates = build_predicates(
                    canon_objects, self.unary, self.binary
                )
                output['pred2gt_cls'] = torch.tensor([
                    self.all_pred_ids[pred] for pred in predicates
                ]).long()
            if 'reg' in self.supervisions:
                targets = build_predicates(
                    canon_objects, ['x(ee, %s)', 'y(ee, %s)', 'z(ee, %s)'],
                    ['x(%s, %s)', 'y(%s, %s)', 'z(%s, %s)']
                )
                output['pred2gt_reg'] = torch.tensor([
                    self.all_target_ids[target] for target in targets
                ]).long()

        return output


class LeonardoLanguageDataset(Dataset):
    def __init__(
        self, data_dir, split, view=1, randview=True, min_crop_ratio=1.,
        normalize_rgb=True, img_size=(224, 224)
    ):
        with open(f'{data_dir}/{split}_nframes.json') as f:
            n_frames = json.load(f)
            self.sequences = list(n_frames.keys())
            n_frames = list(n_frames.values())
            self.cum_n_frames = np.cumsum(n_frames)

    def __getitem__(self, idx):
        data, frame_idx = self.load_h5(idx)
        output = self.get_item(data, frame_idx)

        # Load object IDs
        objects = data['colors'][()].decode().split(',')[:self.n_objects]
        obj_ids = [self.obj_ids[color] for color in objects]
        i = 0
        for _ in range(len(obj_ids), self.n_objects):
            while i in obj_ids:
                i += 1
            obj_ids.append(i)
        canon_objects = [f'object{i:02d}' for i in obj_ids]


def collate(batch):
    batch = {key: [data[key] for data in batch] for key in batch[0]}
    for key in batch:
        if key not in ['action', 'obj_names']:
            batch[key] = torch.stack(batch[key])
    return batch
