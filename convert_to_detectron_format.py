
import pycocotools.mask as mask_util
import cv2
import numpy as np
import os
import json
import glob
import h5py
import argparse

licenses = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 
             'id': 1, 
             'name': 'Attribution-NonCommercial-ShareAlike License'}, 
            {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 
             'id': 2, 
             'name': 'Attribution-NonCommercial License'}, 
            {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 
             'id': 3, 
             'name': 'Attribution-NonCommercial-NoDerivs License'},
            {'url': 'http://creativecommons.org/licenses/by/2.0/', 
             'id': 4, 
             'name': 'Attribution License'},
            {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 
             'id': 5, 
             'name': 'Attribution-ShareAlike License'}, 
            {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 
             'id': 6, 
             'name': 'Attribution-NoDerivs License'}, 
            {'url': 'http://flickr.com/commons/usage/', 
             'id': 7, 
             'name': 'No known copyright restrictions'}, 
            {'url': 'http://www.usa.gov/copyright.shtml', 
             'id': 8, 
             'name': 'United States Government Work'}]
category = [{'supercategory': 'object', 'id': 1, 'name': 'object'}]

IGNORE_PATCH_SIZE = 200

def convert_syn_data_as_video(syn_data_file):
    # load h5 file
    h5_file = h5py.File(syn_data_file, 'r')
    num_videos = len(h5_file['frame0_data'])
    num_frames = len([k for k in h5_file.keys() if k.endswith('_data')])
    videos = []
    annotations = []
    categories = category
    mask_id = 0
    for video_id in range(num_videos):
        video = {
            'id': video_id, 
            'height': h5_file['frame0_data'][video_id].shape[0], 
            'width': h5_file['frame0_data'][video_id].shape[1], 
            'length': num_frames,
            'file_names': [f'v{video_id:06d}_f{frame_id:03d}_data' for frame_id in range(num_frames)],
            'is_syn': True
            }
        videos.append(video)
        
        unique_idx_per_frame = []
        unique_indices = set()
        for frame_idx in range(num_frames):
            # add unique ids in mask
            unique_idx = set(np.unique(h5_file[f'frame{frame_idx}_mask'][video_id]))
            unique_idx.remove(65535)
            unique_idx_per_frame.append(unique_idx)
            unique_indices.update(unique_idx)
        
        masks_cur_video = [h5_file[f'frame{frame_idx}_mask'][video_id] for frame_idx in range(num_frames)]
        # generate annotation info
        
        for unique_id in unique_indices:
            # if unique_id == max(unique_indices):
            #     breakpoint()
            #     continue
            mask_id += 1
            annotation = {
                'id': mask_id, 
                'video_id': video_id, 
                'category_id': 1, 
                'segmentations': [], 
                'areas': [], 
                'bboxes': [],
                'iscrowd': 0,}
            
            exist_checker = False
            for frame_idx, mask in enumerate(masks_cur_video):
                if unique_id == max(unique_idx_per_frame[frame_idx]):
                    area = 0
                else:
                    mask_each_obj = (mask == unique_id).astype(np.uint8)
                    area = mask_each_obj.sum()
                    exist_checker = True
                if area < IGNORE_PATCH_SIZE:
                    annotation['segmentations'].append(None)
                    annotation['areas'].append(None)
                    annotation['bboxes'].append(None)
                else:
                    rle = mask_util.encode(np.array(mask_each_obj[:, :, None], order="F", dtype="uint8"))[0]
                    rle["counts"] = rle["counts"].decode("utf-8")
                    bbox_xywh = list(mask_util.toBbox(rle))
                    annotation['segmentations'].append(rle)
                    annotation['areas'].append(area.astype('float'))
                    annotation['bboxes'].append(bbox_xywh)
            
            if exist_checker:
                annotations.append(annotation)
    json_data = {'info': None, 
                 'licenses': licenses, 
                 'videos': videos, 
                 'annotations': annotations, 
                 'categories': categories}

    save_to_detectron_format(json_data, syn_data_file[:-3]+'_coco.json')

# Handle various number of frameas
def convert_tabletop_data_as_video(syn_data_file, output_fp=None):

    h5_file = h5py.File(syn_data_file, 'r')
    num_videos = h5_file['data'].shape[0]
    num_frames = h5_file['data'].shape[1]
    videos = []
    annotations = []
    categories = category
    mask_id = 0

    for video_id in range(num_videos):
        video = {
            'id': video_id,
            'height': h5_file['data'][video_id,0].shape[0],
            'width': h5_file['data'][video_id,0].shape[1],
            'length': num_frames,
            'file_names': [f'v{video_id:06d}_f{frame_id:03d}_data' for frame_id in range(num_frames)],
            'is_syn': True
            }
        videos.append(video)
    
        unique_indices = set()
        for frame_idx in range(num_frames):
            # add unique ids in mask
            unique_indices.update(np.unique(h5_file['mask'][video_id,frame_idx]))
        unique_indices.remove(65535)

        masks_cur_video = [h5_file['mask'][video_id,frame_idx] for frame_idx in range(num_frames)]
        for unique_id in unique_indices:
            # The tabletop dataset has 0 as the index for the table
            if unique_id == 0:
                continue
            mask_id += 1
            annotation = {
                'id': mask_id,
                'video_id': video_id,
                'category_id': 1,
                'segmentations': [],
                'areas': [],
                'bboxes': [],
                'iscrowd': 0,}

            for frame_idx, mask in enumerate(masks_cur_video):
                mask_each_obj = (mask == unique_id).astype(np.uint8)
                area = mask_each_obj.sum()
                if area < IGNORE_PATCH_SIZE:
                    annotation['segmentations'].append(None)
                    annotation['areas'].append(None)
                    annotation['bboxes'].append(None)
                else:
                    rle = mask_util.encode(np.array(mask_each_obj[:, :, None], order="F", dtype="uint8"))[0]
                    rle["counts"] = rle["counts"].decode("utf-8")
                    bbox_xywh = list(mask_util.toBbox(rle))
                    annotation['segmentations'].append(rle)
                    annotation['areas'].append(area.astype('float'))
                    annotation['bboxes'].append(bbox_xywh)

            annotations.append(annotation)

    json_data = {'info': None,
                 'licenses': licenses,
                 'videos': videos,
                 'annotations': annotations,
                 'categories': categories}

    if output_fp is not None:
        save_to_detectron_format(json_data, output_fp)
    else:
        save_to_detectron_format(json_data, syn_data_file[:-3]+'_coco.json')

        
def convert_real_data_as_video(real_data_root):
    all_videos = [x for x in glob.glob(os.path.join(real_data_root, 'images', '*', '*')) if os.path.isdir(x)]
    all_videos.sort()
    json_data = {'info': None, 'licenses': licenses, 'videos': [], 'annotations': [], 'categories': category}
    videos = []
    annotations = []
    mask_id = 0
    for video_id, video_dir in enumerate(all_videos):
        frames_each_video = [x for x in glob.glob(os.path.join(video_dir, '*.png'))]
        frames_each_video.sort()
        h,w,_ = cv2.imread(frames_each_video[0]).shape
        
        # generate video info
        assert len(frames_each_video[0].split(real_data_root)) == 2, 'path format error'
        relative_frame_path = [x.split(os.path.join(real_data_root, 'images'))[-1] for x in frames_each_video]
        relative_frame_path = [x[1:] if x[0] == '/' else x for x in relative_frame_path]
        masks_each_video = [os.path.join(real_data_root, 'masks', x) for x in relative_frame_path]
        video = {
            'id': video_id, 
            'height': h, 
            'width': w, 
            'length': len(relative_frame_path),
            'file_names': relative_frame_path, 
            'scene': video_dir.split('/')[-2],
            'bin': video_dir.split('/')[-1]}
        videos.append(video)

        # generate annotation info
        for frame_path, mask_path in zip(frames_each_video, masks_each_video):
            assert os.path.basename(frame_path) == os.path.basename(mask_path)

        mask_per_frame = []
        for frame_path, mask_path in zip(frames_each_video, masks_each_video):
            if mask_path.endswith('_0000.png'):
                mask_per_frame.append(np.zeros_like(cv2.imread(frame_path)))
            else:
                mask_per_frame.append(cv2.imread(mask_path))
        mask_per_frame = [x @ [256*256, 256, 1] for x in mask_per_frame]
        # get all unique values in mask_per_frame
        unique_values = np.unique(np.concatenate(mask_per_frame))
        # print(f"scene {video['scene']} bin {video['bin']} has {len(unique_values)-1} objects")
        for unique_value in unique_values:
            if unique_value == 0:
                continue
            mask_id += 1
            annotation = {
                'id': mask_id, 
                'video_id': video_id, 
                'category_id': 1, 
                'segmentations': [], 
                'areas': [], 
                'bboxes': [],
                'iscrowd': 0,}
            existence_flag = False
            for frame_idx, mask in enumerate(mask_per_frame):
                mask_each_obj = (mask == unique_value).astype(np.uint8)
                        
                meta_info = {'image_name': os.path.basename(relative_frame_path[frame_idx]), 'unique_value': unique_value}
                mask_each_obj = remove_isolated_parts(mask_each_obj, meta_info)
                area = mask_each_obj.sum()
                if area < IGNORE_PATCH_SIZE:
                    assert not existence_flag
                    annotation['segmentations'].append(None)
                    annotation['areas'].append(None)
                    annotation['bboxes'].append(None)
                else:
                    existence_flag = True
                    rle = mask_util.encode(np.array(mask_each_obj[:, :, None], order="F", dtype="uint8"))[0]
                    rle["counts"] = rle["counts"].decode("utf-8")
                    bbox_xywh = list(mask_util.toBbox(rle))
                    # bbox_xywh = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1]]
                    # assert(bbox_xywh[2] > 0 and bbox_xywh[3] > 0)
                    annotation['segmentations'].append(rle)
                    annotation['areas'].append(area.astype('float'))
                    annotation['bboxes'].append(bbox_xywh)

            annotations.append(annotation)
    json_data['videos'] = videos
    json_data['annotations'] = annotations
    save_to_detectron_format(json_data, os.path.join(real_data_root, 'video_instances.json'))

def convert_videoannot_to_imageannot(args):
    with open(args.input, 'r') as f:
        vid_annot = json.load(f)

    img_annot = {'info' : vid_annot['info'], 'license' : vid_annot['licenses'], 'categories' : vid_annot['categories']}
    
    images = []
    id_counter = 0
    vid_ids_map = {}
    for i in range(len(vid_annot['videos'])):
        cur_video_meta = vid_annot['videos'][i]
        img_list = []
        vid_ids_map[cur_video_meta['id']] = id_counter
        for frame_idx in range(len(cur_video_meta['file_names'])):
            cur_img = {'id' : id_counter, 'height' : cur_video_meta['height'], 'width' : cur_video_meta['width'],
                        'file_name' : cur_video_meta['file_names'][frame_idx]}
            img_list.append(cur_img)
            id_counter += 1
        images.extend(img_list)
    img_annot['images'] = images
    
    annotations = []
    id_counter = 0
    for annot in vid_annot['annotations']:
        annot_list = []
        for frame_idx in range(len(annot['segmentations'])):
            if annot['segmentations'][frame_idx] is None:
                continue
            cur_annot = {'area' : annot['areas'][frame_idx], 'iscrowd' : annot['iscrowd'], 'id' : id_counter,
                         'category_id' : annot['category_id'], 'bbox' : annot['bboxes'][frame_idx],
                         'segmentation' : annot['segmentations'][frame_idx], 'image_id' : vid_ids_map[annot['video_id']] + frame_idx}
            annot_list.append(cur_annot)
            id_counter += 1
        annotations.extend(annot_list)
    img_annot['annotations'] = annotations

    if args.output is not None:
        save_to_detectron_format(img_annot, args.output)
    else:
        save_to_detectron_format(img_annot, args.input[:-5]+'_perframe.json')

def vid_img_annot_idmap(args):
    with open(args.input, 'r') as f:
        ytvis_json = json.load(f)

    mapping = {}
    cum = 0

    for i, annot in enumerate(ytvis_json['videos']):
        cur_id = annot['id']
        mapping[cur_id] = {}
        mapping[cur_id]['start_id'] = cum
        mapping[cur_id]['length'] = len(annot['file_names'])
        cum += len(annot['file_names'])

    if args.output is None:
        save_to_detectron_format(mapping, os.path.join(os.path.dirname(args.input), "vid_img_idmap.json"))
    else:
        save_to_detectron_format(args.output)

def remove_isolated_parts(mask_each_obj, meta):
    # remove isolated parts with area < IGNORE_PATCH_SIZE
    num_connected_parts, labels = cv2.connectedComponents(mask_each_obj, 4)
    if num_connected_parts>2:
        part_size = [np.sum(labels==x) for x in range(1, num_connected_parts)]
        num_part_large_enough = 0
        for part_idx in range(1, num_connected_parts):
            if np.sum(labels==part_idx) < IGNORE_PATCH_SIZE:
                mask_each_obj[labels==part_idx] = 0
            else:
                num_part_large_enough += 1
        if num_part_large_enough > 1:
            print(f"image {meta['image_name']} has {num_part_large_enough} parts")
            print(part_size)
    return mask_each_obj
                    
def convert_real_data_as_frame(real_data_root):
    all_frame = [x for x in glob.glob(os.path.join(real_data_root, 'images/*/*', '*.png'))]
    all_frame.sort()
    json_data = {'info': {}, 'licenses': licenses, 'images': [], 'annotations': [], 'categories': category}
    images = []
    annotations = []
    mask_id = 0
    for image_id, frame_path in enumerate(all_frame):
        # assert os.path.basename(frame_path) == os.path.basename(mask_path)
        if frame_path.endswith('_0000.png'):
            continue
        
        h,w,_ = cv2.imread(frame_path).shape
        relative_path = frame_path.split(os.path.join(real_data_root, 'images'))[-1]
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
        # parts = relative_path.split('/')
        # scene_id = [x for x in parts if x.startswith('scene')][0]
        # bin_id = [x for x in parts if x.startswith('bin')][0]
        # image_name = os.path.basename(frame_path)
        # relative_path = os.path.join(scene_id, bin_id, image_name)
        image = {
            'license': 1,
            'file_name': relative_path,            
            'height': h,
            'width': w,
            'id': image_id,
            'scene': frame_path.split('/')[-3],
            'bin': frame_path.split('/')[-2]}
        images.append(image)
        mask_path = os.path.join(real_data_root, 'masks', relative_path)
        if mask_path.endswith('_0000.png'):
            mask = np.zeros_like(cv2.imread(frame_path))
        else:
            mask = cv2.imread(mask_path)
        mask = mask @ [256*256, 256, 1]
        
        num_obj = 0
        areas = []
        unique_values = np.unique(mask)
        for unique_value in unique_values:
            if unique_value == 0:
                continue
            mask_id += 1
            annotation = {
                'segmentation': None,
                'area': None,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': None,
                'category_id': 1,
                'id': mask_id,}
            mask_each_obj = (mask == unique_value).astype(np.uint8)
            meta_info = {'image_name': os.path.basename(relative_path), 'unique_value': unique_value}
            mask_each_obj = remove_isolated_parts(mask_each_obj, meta_info)
            area = mask_each_obj.sum()
            if area >= IGNORE_PATCH_SIZE:
                rle = mask_util.encode(np.array(mask_each_obj[:, :, None], order="F", dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                bbox_xywh = list(mask_util.toBbox(rle))
                annotation['segmentation'] = rle
                annotation['area'] = area.astype('float')
                annotation['bbox'] = bbox_xywh
                
                num_obj += 1
                areas = areas + [area]
                annotations.append(annotation)
        print(f"image {relative_path} has {len(unique_values)} uqniue_values and {num_obj} annotations with area {areas}")
                    
    json_data['images'] = images
    json_data['annotations'] = annotations
    save_to_detectron_format(json_data, os.path.join(real_data_root, 'image_instances.json'))

def save_to_detectron_format(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f)
    print('saved to', save_path)

def resize_image(image, new_height, new_width):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

def read_image(image_path):
    image = cv2.imread(image_path)
    return image

def write_image(image, image_path):
    cv2.imwrite(image_path, image)

def resize_dataset(type='image'):
    if type=='image':
        xmem_data_paths = '/home/yili/ObjectEmbedding/Mask2Former/data/AURMR/annotated_real_v1_xmem/*/*/images/*.png'
        new_data_root = '/home/yili/ObjectEmbedding/Mask2Former/data/AURMR/annotated_real_v1_resized/images'
    else:
        xmem_data_paths = '/home/yili/ObjectEmbedding/Mask2Former/data/AURMR/annotated_real_v1_xmem/*/*/masks/*.png'
        new_data_root = '/home/yili/ObjectEmbedding/Mask2Former/data/AURMR/annotated_real_v1_resized/masks'
    origin_data_root = '/home/yili/ObjectEmbedding/Mask2Former/data/AURMR/amazon_real_v1/'
    image_paths = [x for x in glob.glob(xmem_data_paths)]
    image_paths.sort()
    for image_path in image_paths:
        basename = os.path.basename(image_path)
        scene_id = image_path.split('/')[-4]
        bin_id = image_path.split('/')[-3]
        img = read_image(image_path)
        
        origin_image_path = os.path.join(origin_data_root, scene_id, bin_id, basename)
        origin_img = read_image(origin_image_path)
        img = resize_image(img, origin_img.shape[0], origin_img.shape[1])
        
        os.makedirs(os.path.join(new_data_root, scene_id, bin_id), exist_ok=True)
        write_image(img, os.path.join(new_data_root, scene_id, bin_id, basename))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()
    vid_img_annot_idmap(args)
    