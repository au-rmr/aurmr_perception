# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.append('/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/mask2former_frame/demo/../..')
# print(sys.path)
#sys.path.insert(1, os.path.join(sys.path[0], '../..'))
#print(sys.path)
# fmt: on

import tempfile
import time
import warnings
warnings.filterwarnings('ignore')


import cv2
import numpy as np
import tqdm
from PIL import Image


from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config
from mask2former_frame import add_frame_config


from vita import (
    add_vita_config,
)
sys.path.append('/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/mask2former_frame/demo')
from predictor import VisualizationDemo



# constants
WINDOW_NAME = "mask2former demo"

def setup_cfg(config_file,opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vita_config(cfg)
    add_frame_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


        
def is_black_image(image):
	width, height = image.shape
	for x in range(width):
	    for y in  range(height):
	        if image[x,y] != 0:
	           return False
	return True


def main():
    #config_file = '/home/aurmr/workspaces/tejoram/VITA_my/Yi_drive_data/VITA_my-20230117T202349Z-001/VITA_my/frame_R50_v26_c17_csf/frame_R50_v26_c17_csf.yaml'
    config_file = '/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/configs/amazon/frame_R50_v26_c17_csf.yaml'    
    sequence = "/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/data/bin_3F/"
    #frame_names = ['bin_3F_color_0000.png', 'bin_3F_color_0001.png', 'bin_3F_color_0002.png', 'bin_3F_color_0003.png', 'bin_3F_color_0004.png', 'bin_3F_color_0005.png']

    # Get a list of all the files in the directory
    file_list = os.listdir(sequence)

    # Filter the list to only include image files
    image_list = [f for f in file_list if f.endswith('.png') or f.endswith('.jpg')]

    # Sort the list in alphabetical order
    image_list.sort()

    # Generate the frame_names list
    frame_names = [os.path.join(sequence, f) for f in image_list]

    frame_name = " ".join(frame_names)
    output = '/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/vis_real/frame_R50_v26_c17_csf/scene_03_full'
    test_type = 'ytvis'
    opts = ["MODEL.WEIGHTS", "/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/model_final.pth", "DATALOADER.NUM_WORKERS", "0", "INPUT.SAMPLING_FRAME_NUM", len(frame_names)]
    

    mp.set_start_method("spawn", force=True)

    cfg = setup_cfg(config_file, opts)

    demo = VisualizationDemo(cfg, test_type)
    
    

    if sequence:
        # print(sequence[0])
        # if len(sequence) == 1:
        #     sequence = glob.glob(os.path.expanduser(sequence[0]))
        #     assert sequence, "The sequence path(s) was not found"
        sequence_path = sequence
        # for sequence_path in tqdm.tqdm(sequence, disable=not output):
        # if len(frame_name) == 1:
        #     sequence_frame = glob.glob(os.path.join(sequence_path, frame_name[0]))

        #     assert sequence_frame, "The frame name(s) was not found"
        # else:
        sequence_frame = []

        for frame in frame_names:

            if os.path.exists(os.path.join(sequence_path, frame)):
                sequence_frame.append(os.path.join(sequence_path, frame))
            else:
                x = glob.glob(os.path.join(sequence_path, frame))
                assert len(x) == 1, "The frame name(s) was not found"
                sequence_frame.append(x[0])
                        
        sequence_imgs = []

        for frame_path in sequence_frame:

            # use PIL, to be consistent with evaluation
            img = read_image(frame_path, format="BGR")
            sequence_imgs.append(img)
        start_time = time.time()
        predictions = demo.run_on_sequence(sequence_imgs) #, visualized_output, total_pred_labels
        #print(visualized_output)
        #print(total_pred_labels)



        for frame_idx, frame_path in enumerate(sequence_frame):
            if output:
                if os.path.isdir(output):

                    assert os.path.isdir(output), output
                    out_filename = os.path.join(output, os.path.basename(frame_path))
                else:

                    # assert len(args.input) == 1, "Please specify a directory with args.output"
                    os.makedirs(output, exist_ok=True)
                    out_filename = os.path.join(output, os.path.basename(frame_path))
                #visualized_output[frame_idx].save(out_filename)

            else:

                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                #cv2.imshow(WINDOW_NAME, visualized_output[frame_idx].get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break

        new_tensor = []



        if len(frame_names)>1:
            predictions_new = predictions['pred_masks']
        else:
            predictions_new = [predictions[0]["instances"].pred_masks.to("cpu") ] 

        #print(predictions_new)
        x= len(predictions_new)
        y= len(predictions_new[0])

        for i in range (x):
            counter = 0
            for j in range(y):
                mask_name= "mask_{}_{}.png".format(i,j)
                #cv2.imwrite(os.path.join(output,mask_name),np.asarray(predictions_new[i][j])*1024*10)    #Uncomment to visualize each mask
                if not is_black_image(np.asarray(predictions_new[i][j])*1024):
                    temp_j = j

               

            new_tensor.append((predictions_new[i][j]))        
              




        result = np.zeros(new_tensor[0].shape, dtype=np.uint8)           
                
        for i in range(len(new_tensor)):
            result[new_tensor[i] >0] = i+1
            
        #cv2.imwrite(os.path.join(output,"result.png"),result)  
        pil_image = Image.fromarray(result)                                    
        pil_image.save(os.path.join(output,"result.png"))
            
            
