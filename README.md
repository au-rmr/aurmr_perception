# Mask2Former Instance Segmentation

## Installation Steps
1. Create conda environment with Python version 3.9 and then activate it
    ```bash
    conda create --name segnetv2 python=3.9 -y
    conda activate segnetv2
    ````
2. Install appropriate version of pytorch, cudatoolkit and torchvision libraries with GPU support. Along with that also install opencv libraries which will be used for miscellaneous work
    ```bash
    conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
    pip install -U opencv-python
    ```
3. Now clone the detectron repo outside this repo as we only want the libraries to install so after installation you may delete the repo. Follow these steps
    ```bash
    git clone git@github.com:facebookresearch/detectron2.git
    cd detectron2
    pip install -e .
    pip install git+https://github.com/cocodataset/panopticapi.git
    pip install git+https://github.com/mcordts/cityscapesScripts.git
    ```
4. Ninja is yet another build system. It takes as input the interdependencies of files (typically source code and output executables) and orchestrates building them, quickly. So Mask2Former needs ninja, thus clone this repo and install using python
    ```bash
    git clone https://github.com/ninja-build/ninja.git
    cd ninja
    ./configure.py --bootstrap
    ```
5. Install the cudatoolkit-dev which includes cuda-nvcc library which is required by Mask2Former package
    ```bash
    conda install -c conda-forge cudatoolkit-dev
    ```
6. Also we have to install libraries from Mask2Former repo that contains all the network related libraries. Follow these steps
    ```bash
    git clone git@github.com:facebookresearch/Mask2Former.git
    cd Mask2Former
    pip install -r requirements.txt
    cd mask2former/modeling/pixel_decoder/ops
    sh make.sh
    ```

## Important points to note

1. Change the config file path while cloning it to the target machine
2. Change the default parameters as per the filepath of the weights, image path, Number of Sampling fraem number and so on...

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
    	predictions, visualized_output, masks = demo.run_on_image(img)
        for i in masks:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, i)
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
            # pred_scores = predictions["scores"]
            # pred_masks = predictions["pred_masks"]
            # Convert image from OpenCV BGR format to Matplotlib RGB format.
            image = image[:, :, ::-1]
            visualizer = Visualizer(image, self.metadata,
                                    instance_mode=self.instance_mode)
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(
                predictions=instances)
    
            masks = []
            for i in range(len(instances.scores)):
                # cv2.imwrite("/home/soofiyanatar/Documents/AmazonHUB/UIE-main/masks/label_image" +
                #             str(i)+".png", np.asarray(instances.pred_masks)[i])
                masks.append(np.asarray(instances.pred_masks)[i])

            return predictions, vis_output, masks
    ```
