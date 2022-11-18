# Surface Normal method for Grasp Point Detection

## Installation

Before installing libraries for normal based method, install all the installation steps from the branch segnetv2_single_frame and then just install open3d library using the following command

```bash
pip install open3d
```

### Normal STD

This method does not need to train, you can directly inference with the following command:

```python
python inference.py
```

or modify [inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/normal_std/inference.sh) and run `sh inference.sh`
