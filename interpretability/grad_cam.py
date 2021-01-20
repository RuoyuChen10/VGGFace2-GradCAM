# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:37

@author: mick.yi

"""
import numpy as np
import cv2
import torch


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        # self.feature = None
        # self.gradient = None
        # self.net.eval()
        # self.handlers = []
        # self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self.net.cuda()
        self.net.zero_grad()
        output = self.net(inputs.float())  # [1,num_classes]
        if index is None:
            index = torch.argmax(output.cuda())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0]  # [C,H,W]
        weight = torch.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0]  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = torch.sum(cam, axis=0)  # [H,W]
        cam = torch.relu(cam)  # ReLU

        # 数值归一化
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam.cpu().data.numpy(), (224, 224))
        return cam


class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self.net.cuda()
        self.net.zero_grad()
        output = self.net(inputs.float())  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0]  # [C,H,W]
        gradient = torch.relu(gradient)  # ReLU
        indicate = torch.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = torch.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = torch.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[0]  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = torch.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam.cpu().data.numpy(), (224, 224))
        return cam
