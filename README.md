
To add weights and config use this drive link:
https://drive.google.com/drive/folders/1jNwgU3GFHFBISolrBDddTIP-XX-NgYjz?usp=sharing

```python
python mask2former_frame/demo/demo.py --config-file /path/to/configs/amazon/frame_R50_v24_c23_cstf.yaml --sequence /directory/to/data/scene_03/bin_3F/ --frame_name *_color_0000.png *_color_0001.png *_color_0002.png *_color_0003.png *_color_0004.png *_color_0005.png --output /path/to/output/ --test_type ytvis --opts MODEL.WEIGHTS /path/to/weights/model_final.pth DATALOADER.NUM_WORKERS 0 INPUT.SAMPLING_FRAME_NUM 6
```
