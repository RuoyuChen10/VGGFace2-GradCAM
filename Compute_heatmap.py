# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import argparse
import os
import re

import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
import pickle
from PIL import Image
import tensorflow as tf
from numba import cuda as cuda_

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation
from MTCNN_Portable.mtcnn import MTCNN

from tqdm import tqdm

from vggface_models.resnet import *


def get_net(net_name, weight_path=None):
    """
    根据网络名称获取模型
    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """
    if net_name in ['VGGFace2']:
        weight_path = "./checkpoint/resnet50_scratch_weight.pkl"
        net = resnet50(num_classes=8631)
        with open(weight_path, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        net.load_state_dict(weights)
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    return net


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

def prepare_input(img, shape=(224,224)):
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
    img = img.astype(np.float32)-mean_bgr
    img = img.transpose(2, 0, 1)  # C x H x W
    return torch.tensor([img], requires_grad=True),h_start,w_start


def gen_cam(image, img_, mask, box, h_start, w_start, shape=(224,224)):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    #heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    im_shape = img_.shape[:2]
    ratio = float(shape[0]) / np.min(im_shape)
    img_ = cv2.resize(
        img_,
        dsize=(int(np.ceil(im_shape[1] * ratio)),   # width
               int(np.ceil(im_shape[0] * ratio)))  # height
    )
    img_[h_start:h_start+shape[0], w_start:w_start+shape[1]] = 0.5*img_[h_start:h_start+shape[0], w_start:w_start+shape[1]] + 0.5*heatmap
    img_ = cv2.resize(
        img_,
        dsize=(int(np.ceil(im_shape[1])),   # width
               int(np.ceil(im_shape[0])))  # height
    )
    image[
        int(np.floor(box[1]-box[3]*0.15)):int(np.ceil(box[1]+box[3]*1.15)),
        int(np.floor(box[0]-box[2]*0.15)):int(np.ceil(box[0]+box[2]*1.15))] = img_
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, heatmap.astype(np.uint8)

def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.cpu().data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)

def mkdir(name):
    '''创建文件夹'''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

# load input images and corresponding 5 landmarks
def load_img_and_box(img_path, detector):
    #Reading image
    image = Image.open(img_path)
    if img_path.split('.')[-1]=='png':
        image = image.convert("RGB")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    #Detect 5 key point
    face = detector.detect_faces(img)[0]
    box = face["box"]
    image = cv2.imread(img_path)
    return image, box

def box_crop(image, box):
    image = image[
        int(np.floor(box[1]-box[3]*0.15)):int(np.ceil(box[1]+box[3]*1.15)),
        int(np.floor(box[0]-box[2]*0.15)):int(np.ceil(box[0]+box[2]*1.15))]
    return image

def main(args):

    mkdir(args.output_dir)
    # 网络
    net = get_net(args.network, args.weight_path)
    # MTCNN Detector
    with tf.device('gpu:0'):
        detector = MTCNN()
        # 输入
        img,box = load_img_and_box(args.image_path, detector)
        img_ = box_crop(img, box)
    inputs,h_start,w_start = prepare_input(img_, shape=(224,224))
    cuda_.select_device(0)
    cuda_.close()

    # Grad-CAM
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(net, layer_name)
    # GuidedBackPropagation
    gbp = GuidedBackPropagation(net)

    # 输出图像
    image_dict = {}

    mask = grad_cam(inputs.cuda(), args.class_id)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img,img_, mask, box,h_start,w_start)
    grad_cam.remove_handlers()

    inputs.grad.zero_()  # 梯度置零
    grad = gbp(inputs.cpu())

    gb = gen_gb(grad)
    image_dict['gb'] = norm_image(gb)
    # 生成Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict['cam_gb'] = norm_image(cam_gb)

    # 变回原尺寸
    # keys = list(image_dict.keys())
    # for key in keys:
    #     image_dict[key] = cv2.resize(image_dict[key], (224,224))

    save_image(image_dict, os.path.basename(args.image_path), args.network, args.output_dir)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='VGGFace2',
                        help='ImageNet classification network')
    parser.add_argument('--image-path', type=str, default='./test_images/n000002/0001_01.jpg',
                        help='input image path')
    parser.add_argument('--weight-path', type=str, default=None,
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=0,
                        help='class id')
    parser.add_argument('--output-dir', type=str, default='./results/',
                        help='output directory to save results')
    arguments = parser.parse_args()
    torch.backends.cudnn.enabled = True

    torch.backends.cudnn.benchmark = True
    main(arguments)
