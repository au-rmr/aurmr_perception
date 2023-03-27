from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
import argparse
import json
import os


def ytvis_to_coco(args):
    with open(os.path.join(args.output_dir, "results.json"), 'r') as f:
        ytvis_pred = json.load(f)
    
    with open(args.vid_img_idmap, 'r') as f:
        vid_img_idmap = json.load(f)

    cocodt = COCO(args.coco_json)
    image_id_set = set(cocodt.getImgIds())

    prediction = []
    for i, pred in enumerate(ytvis_pred):
        cur_pred = []
        for frame_idx in range(len(pred['segmentations'])):
            if vid_img_idmap[str(pred['video_id'])]['start_id'] + frame_idx not in image_id_set:
                print("not exists")
                continue
            if mask_util.decode(pred['segmentations'][frame_idx]).sum() < 200:
                continue
            cur_pred.append({
                'image_id': vid_img_idmap[str(pred['video_id'])]['start_id'] + frame_idx,
                'category_id': pred['category_id'],
                'segmentation': pred['segmentations'][frame_idx],
                'score': pred['score_perframe'][frame_idx] if 'score_perframe' in pred else pred['score']
            })
        prediction.extend(cur_pred)

    with open(os.path.join(args.output_dir, 'coco_results.json'), 'w') as f:
        json.dump(prediction, f)

def coco_eval(args):
    cocoGt = COCO(args.coco_json)
    cocoDt = cocoGt.loadRes(os.path.join(args.output_dir, 'coco_results.json'))

    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--coco_json', type=str, default="datasets/crop_camera_withlabel/image_instances.json")
    parser.add_argument('--vid_img_idmap', type=str, default="datasets/crop_camera_withlabel/vid_img_idmap.json")
    # parser.add_argument('--coco_json', type=str, default="datasets/amazon_syn/tabletop_syn/test_shard_000000_coco_perframe.json")
    # parser.add_argument('--vid_img_idmap', type=str, default="datasets/amazon_syn/tabletop_syn/vid_img_idmap.json")

    args = parser.parse_args()
    ytvis_to_coco(args)
    coco_eval(args)