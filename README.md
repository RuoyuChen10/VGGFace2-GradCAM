# Face recognition through VGGFace2, renderer from 3DMM face and using GradCAM to find where model interests

> **Author:** Ruoyu Chen
>
> **Updating:** 2021.01.19
>
> **Connect:** cryexplorer@gmail.com

## 1. Environment

Due to model of tf-mesh-render, it needs tensorflow no more 1.12, however we only use tensorflow when bulid 3D model, renderer and MTCNN detection, after that, the focus is on pytorch. So we using tensorflow=1.9. The code **doesn't support CPU version deep learning framework** but is easy to change to CPU version only.

Create environment:

```shell
conda create -n VGGFace2-GradCAM python=3.6
```

Dependent package:

> tensorflow-gpu=1.9
>
> pytorch
>
> opencv-python
>
> pillow
>
> mesh_render

## 2. Download VGGFace2 datasets

The original VGGFace2 datasets come from: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

However, nearly 2020.01.19 I can't visit the website, and can't get the datasets.

Fortunately, one open source website store the datasets in: https://gas.graviti.cn/

You can download VGGFace2 datasets from here: https://www.graviti.cn/open-datasets/VGGFace2

As official described in https://github.com/ox-vgg/vgg_face2: The dataset contains 3.31 million images of 9131 subjects (identities), with an average of 362.6 images for each subject. Images are downloaded from Google Image Search and have large variations in pose, age, illumination, ethnicity and profession (e.g. actors, athletes, politicians). The whole dataset is split to a training set (including **8631** identities) and a test set (including **500** identities).

So there is **8631** identities in training set and we will using it for face recognition classes when training.

## 3. Download Pytorch version VGGFace2 face recognition network

The work refers: https://github.com/cydonia999/VGGFace2-pytorch, that's not the official implement.

You can git clone the repo, but we only using the `models/resnet.py` and `models/senet.py`

When get the datasets of training set called `VGGFace2_vggface2_train.tar.gz` and the model.

First download the pre-trained model, the model is convert from coffee version from official file, don't worry it doesn't work.

| arch_type          |                        download link                         |
| :----------------- | :----------------------------------------------------------: |
| `resnet50_ft`      | [link](https://drive.google.com/open?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU) |
| `senet50_ft`       | [link](https://drive.google.com/open?id=1YtAtL7Amsm-fZoPQGF4hJBC9ijjjwiMk) |
| `resnet50_scratch` | [link](https://drive.google.com/open?id=1gy9OJlVfBulWkIEnZhGpOLu084RgHw39) |
| `senet50_scratch`  | [link](https://drive.google.com/open?id=11Xo4tKir1KF8GdaTCMSbEQ9N4LhshJNP) |

Download model is `pkl` files.

You can try this to test:

```python
import torch
import pickle
from models.resnet import *	# the resnet model
import numpy as np
import cv2

# Image preprocessing method
def precessing(img, shape):
    '''
    img: the input image
    shape: resize shape
    '''
    mean_bgr = (131.0912, 103.8827, 91.4953)  # from resnet50_ft.prototxt
    im_shape = img.shape[:2]
    ratio = float(shape[0]) / np.min(im_shape)
    img = cv2.resize(
        img,
        dsize=(int(np.ceil(im_shape[1] * ratio)),   # width
               int(np.ceil(im_shape[0] * ratio)))  # height
    )
    new_shape = img.shape[:2]
    h_start = (new_shape[0] - shape[0])//2
    w_start = (new_shape[1] - shape[1])//2
    img = img[h_start:h_start+shape[0], w_start:w_start+shape[1]]
    cv2.imwrite("result.jpg",img)
    img = img.astype(np.float32)-mean_bgr
    img = img.transpose(2, 0, 1)  # C x H x W
    return img

# the box
def box_crop(image, box):
    image = image[
        int(np.floor(box[1]-box[3]*0.15)):int(np.ceil(box[1]+box[3]*1.15)),
        int(np.floor(box[0]-box[2]*0.15)):int(np.ceil(box[0]+box[2]*1.15))]
    return image

weight_path = ".../resnet50_scratch_weight.pkl"	# path to your pretrained file
model = resnet50(num_classes=8631)	# choose the truth model
# load the weight to model
with open(weight_path, 'rb') as f:
    obj = f.read()
weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
model.load_state_dict(weights)
# CUDA
model.cuda()
# Eval
model.eval()

# try different images, the box is giving.
image_path = ".../VGGFace2/train/n000006/0001_01.jpg"
box = [78, 49, 121, 153]

# image_path = ".../VGGFace2/train/n000006/0004_06.jpg"
# box = [62, 54, 94, 120]

# image_path = ".../VGGFace2/train/n000002/0001_01.jpg"
# box = [161, 147, 226, 317]

# image_path = ".../VGGFace2/train/n000002/0011_01.jpg"
# box = [39, 82, 126, 156]

image = cv2.imread(image_path)
image = box_crop(image,box)
image_ = precessing(img=image, shape=(224,224))
image_input = torch.tensor([image_], requires_grad=True)
y = model(image_input.cuda().float())

print(y.argmax(dim=1))
# that will get the ID index
```

