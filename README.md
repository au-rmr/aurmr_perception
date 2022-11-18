# Mask2Former Instance Segmentation

## Installation Steps
1. 

## Important points to note

1. Change the config file path while cloning it to the target machine
2. 

## Changes for single frame detection

1. Added argument for taking single input image file along with its path
    
    ```python
    parser.add_argument("--image_input", 
    default="/home/soofiyanatar/Documents/AmazonHUB/UIE-main/annotated_real_v1_resized/images/scene_03/bin_1E/bin_1E_color_0006.png",
    help="Path to Image file.")
    ```
    
2. Added this line instead of **`demo = VisualizationDemo(cfg, test_type=args.test_type)`**
    
    ```python
    demo = VisualizationDemo(cfg)
    ```
    
1. Added this block which will be used for single image detection
    
    ```python
    if args.image_input:
    	img = read_image(args.image_input, format="BGR")
    	predictions, visualized_output = demo.run_on_image(img)
    	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    	cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    	cv2.waitKey(0)  # esc to quit
    ```
    
2. In the initialization of the `VisualizationDemo()` class, I commented on these lines, because we don't want to use these datasets, as we are using coco which is used for frame detection
    
    ```python
       # if test_type == 'coco': 
       #     self.predictor = FramePredictor(cfg)
       # elif test_type == 'ytvis':
       #     self.predictor = VideoPredictor(cfg)
    ```
    
    and added these lines, where I am adding the default processing and parallel processing
    
    ```python
    if parallel:
        num_gpu = torch.cuda.device_count()
        self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
    else:
        self.predictor = DefaultPredictor(cfg)
    ```
    
3. Uncommented this block `raise NotImplementedError("Use `run_on_sequence` instead.")` which is in the `run_on_image()` function, and added a for loop for saving all the images with individual masks.
    
    ```python
    def run_on_image(self, image):
            """
            Args:
                image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                    This is the format used by OpenCV.
            Returns:
                predictions (dict): the output of the model.
                vis_output (VisImage): the visualized image output.
            """
            # raise NotImplementedError("Use `run_on_sequence` instead.")
            vis_output = None
            predictions = self.predictor(image)
            print(predictions)
            # pred_scores = predictions["scores"]
            # pred_masks = predictions["pred_masks"]
            # Convert image from OpenCV BGR format to Matplotlib RGB format.
            image = image[:, :, ::-1]
            visualizer = Visualizer(image, self.metadata,
                                    instance_mode=self.instance_mode)
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(
                predictions=instances)
    
            for i in range(len(instances.scores)):
                cv2.imwrite("/home/soofiyanatar/Documents/AmazonHUB/UIE-main/masks/label_image" +
                            str(i)+".png", np.asarray(instances.pred_masks)[i])
    
            return predictions, vis_output
    ```
