import cv2
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# %matplotlib inline

coco = COCO(
    '/home/soofiyanatar/Downloads/Full_Depth_image(1).json')
img_dir = '/home/soofiyanatar/datasets/Full_Dataset/'
for image_id in range(1):
    image_id = 25
    img = coco.imgs[image_id]
    image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
    # plt.imshow(image, interpolation='nearest')
    # plt.show()

    # cat_ids = coco.getCatIds()
    # anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    # anns = coco.loadAnns(anns_ids)
    # mask = coco.annToMask(anns[0])
    # print(len(anns))
    # counter = 1
    # for i in range(len(anns)):
    #     print(anns[i])
    #     mask += coco.annToMask(anns[i])*counter*40+100

    # plt.imshow(mask)
    # plt.show()
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    anns_img = np.zeros((img['height'], img['width']))
    counter = 1
    for ann in anns:
        anns_img = np.maximum(anns_img, coco.annToMask(
            ann)*ann['category_id'])*counter*10
        counter += 1
        cv2.imwrite(
            f"/home/soofiyanatar/datasets/Full_Dataset/labelf_1.png", anns_img)
    # gray = cv2.cvtColor(mask, cv2.COLOR_BW2GRAY)
    # plt.imshow(mask)
    # plt.show()
    img = cv2.imread("/home/soofiyanatar/datasets/Full_Dataset/rgb9.png")
    cv2.imwrite("/home/soofiyanatar/datasets/Full_Dataset/rgb9.png", img)

    # cv2.waitKey(0)


##############################################################################################################################################################
# import cv2

# img_dir = '/home/soofiyanatar/datasets/Full_Dataset/'

# img = cv2.imread("/home/soofiyanatar/datasets/Full_Dataset/rgb.png")
# img1 = cv2.imread("/home/soofiyanatar/datasets/Full_Dataset/depth.png")
# img2 = cv2.imread("/home/soofiyanatar/datasets/Full_Dataset/rgb_empty.png")
# img3 = cv2.imread("/home/soofiyanatar/datasets/Full_Dataset/depth_empty.png")

# img = img[1520:2000, 1955:2595]
# img1 = img1[1520:2000, 1955:2595]
# img2 = img2[1520:2000, 1955:2595]
# img3 = img3[1520:2000, 1955:2595]

# # img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
# # img1 = cv2.resize(img1, (640, 480), interpolation = cv2.INTER_AREA)
# # img2 = cv2.resize(img2, (640, 480), interpolation = cv2.INTER_AREA)
# # img3 = cv2.resize(img3, (640, 480), interpolation = cv2.INTER_AREA)
# # depth = cv2.imread("/home/soofiyanatar/rgb_2.png")
# # label = cv2.imread("/home/soofiyanatar/rgb_3.png")
# # cv2.imshow("i",img)
# # cv2.waitKey(0)
# # img = img[176:315, 32:371]
# # depth = depth[176:315, 32:371]
# # label = label[176:315, 32:371]
# # dim = (1280, 720)
# # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# # depth = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# # label = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# cv2.imwrite("/home/soofiyanatar/Downloads/rgb_16.png", img)
# cv2.imwrite("/home/soofiyanatar/Downloads/depth_16.png", img1)
# cv2.imwrite("/home/soofiyanatar/Downloads/rgb_e.png", img2)
# cv2.imwrite("/home/soofiyanatar/Downloads/depth_e.png", img3)
# # cv2.imwrite("/home/soofiyanatar/rgb_2.png", depth)
# # cv2.imwrite("/home/soofiyanatar/rgb_3.png", label)

# 33

# import cv2

# img = cv2.imread("/home/soofiyanatar/Downloads/rgb_16.png")
# img1 = cv2.imread("/home/soofiyanatar/Downloads/data/depth-input/000004-1.png")

# for i in img:
#     print(i)


img = cv2.imread("/home/soofiyanatar/datasets/Full_Dataset/rgb_2.png.png")
# img = cv2.resize(img, (360, 160), interpolation=cv2.INTER_AREA)
img = img[1680:1840, 2095:2455]
cv2.imwrite("/home/soofiyanatar/datasets/Full_Dataset/rgb_segnet_1.png", img)
img = cv2.imread("/home/soofiyanatar/datasets/Full_Dataset/depth_2.png", 0)
img = img[1680:1840, 2095:2455]
cv2.imwrite("/home/soofiyanatar/datasets/Full_Dataset/depth_segnet_1.png", img)
